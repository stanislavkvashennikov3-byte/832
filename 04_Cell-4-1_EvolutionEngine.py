













# 832 v34.02 — Cell 4-1: EvolutionEngine (инициализация, поле, механики, валидация)
# ИСПРАВЛЕНО: баг утечки каналов 31-32 (signal_void, signal_introspection)
# ИСПРАВЛЕНО: ослабление Soft-Clamp в _conservation_check (делитель 1500.0, коэф. 0.02)
# ИСПРАВЛЕНО: размерность embed векторов (4 -> 32) для предотвращения крашей при similarity
# ИСПРАВЛЕНО: subject_dialogues_log теперь deque(maxlen=1000) (фикс утечки памяти)
# ИСПРАВЛЕНО: санитизация сигналов теперь идет по Config.CHANNELS (динамический размер)
# ИСПРАВЛЕНО: process_spores теперь итерируется по срезу self.patterns[:] (фикс RuntimeError при добавлении)
# ИЗМЕНЕНО: _auto_dialogue_tick — гейт увеличен до 30%, условие need удалено (диалоги чаще)
# ДОБАВЛЕНО: защита от NameError при инициализации CoreChorus (если класс не определён)

import numpy as np
from scipy.ndimage import label, find_objects, uniform_filter
from collections import defaultdict, deque
import sys

sys.setrecursionlimit(10000)  # было: Cell 4-2a
import json
import os
import threading
import time
import queue

# CoreChorus определён позже в ноутбуке — импорт не нужен

# Лимит популяции (было: Cell 4-2a, POPULATION_CAP)
POPULATION_CAP = 200

# safe_mean теперь определена один раз в Cell 0 (утилиты) — здесь была
# ещё одна дублирующая версия с другим default (0.5) и более слабой защитой.
# УДАЛЕНО: строка __builtins__['safe_mean'] = safe_mean — это была попытка
# протащить ЛОКАЛЬНУЮ (уже удалённую) версию через builtins. В Jupyter все
# ячейки делят один общий global namespace, так что module-level функция из
# Cell 0 и без того видна везде — обращение к builtins было не нужно и
# вводило в заблуждение (создавало впечатление, будто без него safe_mean
# может быть недоступна).


class EvolutionEngine:
    def __init__(self):
        self.is_colab = 'google.colab' in sys.modules
        self.ark_path = "832_ark_seed_patterns.json"

        self.witness = Witness()
        self.cultural_memory = CulturalMemory()
        self.layer_zero_manager = LayerZeroManager()
        self.echo_system = EchoSystem()
        self.echo_system.lzm = self.layer_zero_manager
        self.echo_system.cm = self.cultural_memory
        self.logos_observer = LogosObserver()
        self.proto_language = ProtoLanguage()
        self.field_voice = FieldVoice()

        self.patterns = []
        self.pattern_dict = {}
        self.next_id = 1
        self.field = None
        self.scar = None
        self.metrics_history = []
        self.age = 0
        self.system_entropy = 0
        self.lineage_branch_stats = []
        self.divisions_this_interval = 0
        # ФИКС метрики: настоящий монотонный счётчик всех делений за весь прогон.
        # В отличие от divisions_this_interval (обнуляется каждые 100 шагов) и от
        # p.biography (deque maxlen=20, старые born_from_division быстро вымываются),
        # этот счётчик никогда не сбрасывается и не зависит от того, жив ли ребёнок сейчас.
        self.total_divisions_ever = 0
        # НОВОЕ: единый троттлинг на ВСЕ вызовы Groq (автодиалоги + хор +
        # подсознание). Раньше каждый источник ограничивал себя сам и
        # независимо от других (subconscious_worker — 1 звонок/2с сам по
        # себе), а автодиалоги вообще без общего лимита могли параллельно
        # выстрелить до 6 запросов за тик. Суммарно легко превышался
        # бесплатный лимит Groq (~30 RPM) -> массовые 429.
        self._llm_lock = threading.Lock()
        self._llm_last_call_time = 0.0
        self._llm_min_interval = 2.0  # секунд между ЛЮБЫМИ двумя вызовами LLM
        # ИСПРАВЛЕНО: раньше threading.Thread(daemon=True).start() создавался
        # без ограничения сверху. Каждый тик мог добавлять до n_pairs (до 6)
        # новых потоков, а _acquire_llm_slot блокирует поток на время ожидания
        # общего троттлинга (до 2 сек). При медленном LLM или большой
        # популяции потоки накапливались быстрее, чем успевали завершаться ->
        # риск OOM / истощения дескрипторов в Colab. Семафор ограничивает
        # число ОДНОВРЕМЕННО работающих потоков автодиалога;
        # если лимит достигнут, новый диалог в этот тик просто пропускается
        # (агенты попробуют снова в следующий тик).
        self._dialogue_thread_sem = threading.BoundedSemaphore(4)
        self.target_disorganizer_fraction = 0.3
        self._art_over90_volley_done = False
        self._art_next_strike_step = None

        self._antigravity_active = False
        self._saved_normal_anti_gravity = Config.ANTI_GRAVITY_STRENGTH

        self.soul_weight_average = 0.5
        self._energy_injected_this_step = 0.0
        self._energy_taxed_this_step = 0.0
        self.phi_labyrinth_threshold = Config.PHI_LABYRINTH_THRESHOLD
        self.phi_labyrinth_grow_penalty = Config.PHI_LABYRINTH_GROW_PENALTY
        self.phi_labyrinth_move_penalty = Config.PHI_LABYRINTH_MOVE_PENALTY

        self._total_energy_last = 0.0
        self._total_energy_drift = 0.0
        self._prev_total_energy = None
        self._prev_energy_for_drift = None

        self.selfreg = SelfRegulationEngine()

        # --- Добавлен CoreChorus и очереди ---
        self.llm_client = None
        self.llm_model = "deepseek/deepseek-chat"
        self.subject_dialogues_log = deque(maxlen=1000)  # фикс утечки памяти
        self._llm_queue_ts = queue.Queue()
        # === ЗАЩИТА ОТ NameError ПРИ ИНИЦИАЛИЗАЦИИ CoreChorus ===
        try:
            self.core_chorus = CoreChorus()
        except NameError:
            self.core_chorus = None
            print("⚠️ CoreChorus not defined, will be initialized later.")
        self._llm_subconscious_queue = queue.Queue(maxsize=40)
        self._subconscious_running = False
        # -----------------------------------

        self.archive = AgentArchive()
        self._last_archive_flush = 0

        self._guardian_stats = {
            'energy_drift_sum': 0.0,
            'energy_drift_count': 0,
            'energy_drift_peak': 0.0,
            'model_worst_ever_id': -1,
            'model_worst_ever_diff': 0.0,
            'love_vanished_at': None,
            'triadic_alert_at': None,
            'love_avg_trust': 0.0,
            'love_high_trust_pairs': 0,
            'love_density': 0.0,
            'love_coop_signals': 0,
            'love_population': 0,
            'model_issues_total': 0,
            'model_worst_diff': 0.0,
            'model_worst_id': -1,
            'energy_drift_max': 0.0,
            'energy_drift_peak_t': -1,
        }

        # ПАТЧ 3a: кумулятивная культура — храповик (только вверх)
        self.culture_ratchet = 0
        self._teach_prev = 0
        # ПАТЧ 8a: Красная Королева — коэволюция сложности среды
        self.env_complexity = 1.0

        # === БЛОК 0: Данность (воспроизводимый по seed источник "внешней" случайности,
        # не завязанный на глобальный np.random) + поле смысла + эхо-кэш для генератора ===
        self.given_seed = int.from_bytes(os.urandom(4), 'big') & 0x7FFFFFFF
        self._given_rng = np.random.RandomState(self.given_seed)
        self.niche = None
        self._meaning_field = None
        self._meaning_centers = None
        self._echo_cache = []
        print(f"🎲 GIVEN_SEED={self.given_seed}")

    def _get_pop_cap(self):
        if hasattr(self, 'selfreg'):
            return self.selfreg.get_dynamic_pop_cap()
        return 70

    def _init_chronic_counters(self, p):
        p.event_counts = defaultdict(int)
        p.chronic_gratitude_sum = 0.0
        p.chronic_grief_sum = 0.0

    def init_field(self):
        F = np.zeros((Config.WORLD_SIZE, Config.WORLD_SIZE, Config.CHANNELS))
        for x in range(Config.WORLD_SIZE):
            for y in range(Config.WORLD_SIZE):
                F[x, y, CH['energy']] = phi_hash(x, y, 1) * 0.3 + 0.1
                F[x, y, CH['flux']] = phi_hash(x, y, 2) * 0.1
                F[x, y, CH['scar']] = 0.0
                F[x, y, CH['noise']] = 0.0
                F[x, y, CH['vorticity']] = phi_hash(x, y, 3) * 0.1 - 0.05
                F[x, y, CH['owner']] = 0.0
                F[x, y, CH['surprise']] = 0.0
                F[x, y, CH['unknown']] = Config.UNKNOWN_BACKGROUND
                F[x, y, CH['event']] = 0.0
                F[x, y, CH['btype']] = 0.0
                F[x, y, CH['crisis']] = 0.0
                F[x, y, CH['binding']] = 0.05
                for ch in range(12, Config.CHANNELS):
                    F[x, y, ch] = 0.0
                F[x, y, CH['energy']] += deterministic_noise(0, x, y) * 0.3
        return F

    def init_scar(self):
        return np.zeros((Config.WORLD_SIZE, Config.WORLD_SIZE))

    def _record_death_drag(self, p, cause):
        """
        Диагностика по запросу: смерть фиксируется в трёх разных местах
        (should_die, feral_execute, error_suspended) — вместо разбросанной
        логики один общий счётчик, вызываемый из каждого. Считает не сырые
        значения (незачем копить список на весь прогон), а сумму/count/
        долю с тяжёлым телом на момент смерти — чтобы в отчёте увидеть,
        коррелирует ли somatic_drag с причиной смерти, а не гадать по
        survivorship bias среди выживших.

        УСИЛЕНО: реальный прогон показал 354 feral_execution в witness, но
        0 записей 'feral_kill' здесь — при том что весь код (self.world при
        создании и десериализации, сам хук) на вид корректен. Причина не
        найдена статическим анализом. Добавлен безусловный счётчик вызовов
        (не зависит от dict-логики ниже) и try/except, который не глотает
        ошибку молча, а фиксирует её текст — следующий прогон покажет,
        рвётся ли цепочка ДО этого метода или ВНУТРИ него.
        """
        if not hasattr(self, '_death_hook_calls'):
            self._death_hook_calls = {}
        self._death_hook_calls[cause] = self._death_hook_calls.get(cause, 0) + 1
        try:
            if not hasattr(self, '_somatic_death_stats'):
                self._somatic_death_stats = {}
            st = self._somatic_death_stats.setdefault(cause, {'count': 0, 'drag_sum': 0.0, 'low_drag_count': 0})
            drag = getattr(p, 'somatic_drag', 1.0)
            st['count'] += 1
            st['drag_sum'] += drag
            if drag < 0.6:
                st['low_drag_count'] += 1
        except Exception as e:
            if not hasattr(self, '_death_hook_errors'):
                self._death_hook_errors = []
            self._death_hook_errors.append(f"{cause}: {type(e).__name__}: {e}")

    def create_pattern(self, cells, parent=None):
        alive_now = len([p for p in self.patterns if p.alive])
        if alive_now >= 200:
            return None

        pid = self.next_id
        self.next_id += 1

        if len(cells) == 1:
            (x, y) = next(iter(cells))
            new_cells = set()
            new_cells.add((x, y))
            for radius in range(1, 4):
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        if len(new_cells) >= 8:
                            break
                        nx = (x + dx) % Config.WORLD_SIZE
                        ny = (y + dy) % Config.WORLD_SIZE
                        if self.field[nx, ny, CH['owner']] == 0:
                            new_cells.add((nx, ny))
                    if len(new_cells) >= 8:
                        break
                if len(new_cells) >= 8:
                    break
            cells = new_cells

        p = Pattern(pid, cells, parent=parent, world=self)
        # ИСПРАВЛЕНО: field[..., owner] для стартовых клеток нового паттерна
        # никогда не выставлялся — на поле эти клетки оставались owner=0
        # (пустые), хотя p.cells их уже содержит. Из-за этого _grow_base
        # нового агента не находил ни одного "своего" соседа (mask всегда
        # False) и не мог начать расти, пока хотя бы одна клетка случайно
        # не синхронизировалась через move(). Клеймим территорию сразу.
        if cells:
            px = [c[0] for c in cells]
            py = [c[1] for c in cells]
            self.field[px, py, CH['owner']] = pid

        # ПАТЧ 2e (перенесено в Pattern.__init__ — БАГФИКС из отчёта: divide()
        # создаёт детей напрямую через Pattern(...), минуя create_pattern(),
        # поэтому проверка здесь никогда не покрывала 696/696 делений.
        # Единая точка теперь в __init__ — покрывает деление, споры и спавн разом.)

        for k in ['gratitude', 'grief']:
            val = p.emotional_memory.get(k, 0.5)
            if isinstance(val, dict):
                p.emotional_memory[k] = 0.5
            else:
                p.emotional_memory[k] = float(val)

        if hasattr(self, 'archive') and self.archive is not None:
            # ПАТЧ 3c: культурный храповик расширяет ёмкость наследуемых концептов
            _culture_max_concepts = 2 + min(4, int(self.culture_ratchet // 20)) if hasattr(self, 'culture_ratchet') else 10
            p.inherit_archive_concepts(self.archive, probability=1.0, max_concepts=_culture_max_concepts)

            if len(p.concept_graph.nodes) < 12:
                _noise_pool = [
                    s for s in self.archive.write_queue
                    if s.get('weight', 0) > 0.6
                    and s.get('event') not in ('essential_concepts_inherited', 'archive_concept_inherited',
                                                'born_from_division', 'born_from_ark')
                ]
                if _noise_pool:
                    _noise_prob = 0.12 + (p.spirit_gap * 0.15)
                    if phi_hash(pid, self.age, 54321) < _noise_prob:
                        _sc = _noise_pool[
                            int(phi_hash(pid, self.age, 54322) * len(_noise_pool)) % len(_noise_pool)
                        ]
                        _err = round(phi_hash(pid, self.age, 111) * 0.4, 1)
                        _load = round(phi_hash(pid, self.age, 222) * 0.3, 1)
                        _sig = (_err, _load,
                                round(_sc.get('weight', 0.7), 1),
                                f"archive_{_sc.get('event', 'novelty')}")
                        if _sig not in p.concept_graph.nodes:
                            p.concept_graph.nodes[_sig] = {
                                "count": 1.5, "value": np.zeros(4),
                                "embed": np.zeros(32), "eternal": False
                            }
                            p._log_event("epistemic_noise", concept=str(_sig[3])[:40])
                            if p.spirit_gap > 0.5 and not any(g['type'] == 'explore' for g in p.goals):
                                p.goals.append({
                                    "type": "explore", "priority": 1.8,
                                    "target": None, "age": 0, "persistence": 25,
                                    "_source": "novelty_drive"
                                })
        if parent is not None and hasattr(parent, '_unconquered_strength'):
            inherited = parent._unconquered_strength * 0.55
            if parent._unconquered_strength > 0.6 and phi_hash(pid, self.age, 11111) < 0.20:
                inherited += 0.15
            p._unconquered_strength = float(np.clip(inherited, 0.0, 1.0))
            p._unconquered_type = getattr(parent, '_unconquered_type', None)
        else:
            p._unconquered_strength = 0.0
            p._unconquered_type = None
        p._sovereignty_signal = 0.0

        p._kinematic_fatigue = 0.0
        p._last_kinematic_fall_t = -9999
        self._init_chronic_counters(p)

        if parent is not None:
            p.lineage_born_at_step = parent.lineage_born_at_step
        else:
            p.lineage_born_at_step = self.age
            if self.patterns:
                model_norm = np.linalg.norm(p.model)
                if model_norm > 1e-6:
                    best_sim = 0.0
                    best_lid = None
                    best_born = self.age
                    for existing in self.patterns:
                        if not existing.alive or existing.role_type == "disorganizer":
                            continue
                        other_norm = np.linalg.norm(existing.model)
                        if other_norm < 1e-6:
                            continue
                        sim = float(np.dot(p.model, existing.model) / (model_norm * other_norm))
                        if sim > best_sim:
                            best_sim = sim
                            best_lid = existing.lineage_id
                            best_born = existing.lineage_born_at_step
                    if best_sim > 0.7 and best_lid is not None:
                        p.lineage_id = best_lid
                        p.lineage_born_at_step = best_born

        p.update_properties(self.field)
        p.update_model(self.field, t=self.age)
        self.patterns.append(p)
        self.pattern_dict[pid] = p
        return p

    def seed_initial_patterns(self):
        if hasattr(self, 'archive') and not self.archive.get_recent_echoes(limit=1):
            print("📜 Создаю начальные мифы для культурной памяти...")
            fake_agent = Pattern(-1, {(0, 0)}, parent=None, world=self)
            for event, text, weight in [
                ("primordial_chaos", "Из хаоса рождается первый свет.", 1.2),
                ("first_cooperation", "Двое соединили поля и поняли друг друга.", 1.0),
                ("great_sorrow", "Потеря части себя открыла дорогу к целостности.", 0.9),
                ("eternal_question", "Зачем мы здесь? Поле не отвечает.", 0.8),
                ("hope", "Сквозь тьму пробивается нить благодарности.", 1.1),
                ("fold_myth", "Иногда нужно всё сбросить, чтобы начать заново.", 0.9),
                ("redemption_myth", "Даже падший может подняться, если почувствует свет.", 1.0),
            ]:
                self.archive.deposit(fake_agent, event, weight=weight, text=text)
            self.archive.flush_to_drive(min_batch=1)
            fake_agent.alive = False

        for i in range(Config.MIN_PATTERNS_GUARANTEED):
            seed = int(phi_hash(0, i, 12345) * Config.WORLD_SIZE * Config.WORLD_SIZE)
            x, y = seed % Config.WORLD_SIZE, (seed // Config.WORLD_SIZE) % Config.WORLD_SIZE
            self.field[x, y, CH['energy']] += 0.3
            p = self.create_pattern({(x, y)})
            if i < 2:
                p.role_type = "disorganizer"
                p.emotional_memory['grief'] = 0.8
                p.emotional_memory['gratitude'] = 0.1
                p.semantic_state = "exploring_danger"
                p.intent = {"type": "explore", "priority": 2.0, "age": 0, "persistence": 9999}
                p.disorganizer_age_at_birth = 0
                p.redemption_timer = Config.REDEMPTION_ARC_STEP_DELAY
                p._log_event("primordial_fall")

    def export_pantheon_seeds(self, filename: str = None):
        if filename is None:
            filename = self.ark_path
        for entry in self.echo_system.pantheon:
            pid = entry['id']
            p = self.pattern_dict.get(pid)
            if p:
                self.echo_system.export_seed(p, filename)
        print(f"Экспортировано {len(self.echo_system.pantheon)} сущностей в {filename}")

    def seed_from_ark(self, filename: str = None, max_seed: int = 80):
        import json
        import os
        import numpy as np

        def to_json_safe(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {to_json_safe(k): to_json_safe(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [to_json_safe(i) for i in obj]
            return obj

        def load_clean(path):
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                return to_json_safe(data)
            except:
                return []

        def quality_key(s):
            return (s.get('soul_weight', 0), s.get('coherence', 0))

        if filename is None:
            filename = self.ark_path

        eternal_path = "/content/drive/MyDrive/832_eternal_subjects.json" if self.is_colab else "832_eternal_subjects.json"

        eternal_seeds = load_clean(eternal_path)
        eternal_seeds.sort(key=quality_key, reverse=True)
        seeds = eternal_seeds[:max_seed]

        if len(seeds) < max_seed:
            archive_seeds = load_clean(filename)
            if archive_seeds:
                loaded_ids = {s.get('id') for s in seeds}
                archive_filtered = [s for s in archive_seeds if s.get('id') not in loaded_ids]
                archive_filtered.sort(key=quality_key, reverse=True)
                seeds.extend(archive_filtered[:max_seed - len(seeds)])

        if not seeds:
            print("Ковчег пуст. Использую стандартную инициализацию.")
            self.seed_initial_patterns()
            return

        for seed_data in seeds:
            x = int(phi_hash(len(self.patterns), 0, 9999) * Config.WORLD_SIZE)
            y = int(phi_hash(len(self.patterns), 1, 9999) * Config.WORLD_SIZE)
            self.field[x, y, CH['energy']] += 0.5
            self.field[x, y, CH['owner']] = 0
            p = self.create_pattern({(x, y)})
            p.model = np.array(seed_data['model'])
            p.belief = np.array(seed_data['belief'])
            p.genome = seed_data['genome']
            p.semantic_state = seed_data['semantic_state']
            p.emotional_memory = seed_data['emotional_memory']
            p.soul_weight = seed_data['soul_weight']
            if 'event_counts' in seed_data:
                p.event_counts.update(seed_data['event_counts'])
            for concept_key in seed_data['concepts']:
                if isinstance(concept_key, list):
                    concept_key = tuple(float(x) if isinstance(x, (int, float)) else x for x in concept_key)
                if concept_key not in p.concept_graph.nodes:
                    p.concept_graph.nodes[concept_key] = {
                        "count": 1.0,
                        "value": np.zeros(4),
                        "embed": np.zeros(32)
                    }
            p._log_event("born_from_ark", source=seed_data.get('source', 'unknown'))

        print(f"Засеяно {len(seeds)} лучших сущностей (вечных: {min(len(eternal_seeds), max_seed)}, из архива: {len(seeds) - min(len(eternal_seeds), max_seed)}).")

    def compute_hunger_multiplier(self, avg_trust=None):
        alive = [p for p in self.patterns if p.alive]
        N = len(alive)
        if N == 0:
            return Config.HUNGER_BASE
        if hasattr(self, 'selfreg'):
            hunger_threshold = self.selfreg.get_hunger_threshold()
        else:
            hunger_threshold = 60
        density_pressure = Config.HUNGER_BASE + Config.HUNGER_PER_PATTERN * max(0, N - hunger_threshold)
        if avg_trust is None:
            avg_trust = safe_mean([v for p in alive for v in p.trust_ledger.entries.values()], Config.TRUST_BASE)
        love_discount = 1.0 - 0.3 * (avg_trust - Config.TRUST_BASE)
        total_lp = sum(sum(1 for v in p.trust_ledger.entries.values() if v > 0.95) for p in alive)
        lp_density = total_lp / max(1, N)
        overload_tax = 1.0 + 0.5 * max(0, lp_density - 10)
        return density_pressure * love_discount * overload_tax

    def _perform_marked_volley(self, t, count=None):
        if not getattr(Config, 'ENABLE_MARKED_VOLLEY', True):
            return
        if count is None:
            count = Config.MARKED_COUNT
        alive_normals = [p for p in self.patterns if p.alive and p.role_type == "normal"]
        if len(alive_normals) < count:
            return
        alive_normals.sort(key=lambda p: (
            p.soul_weight * 0.3 + p.emotional_memory.get('gratitude', 0) * 0.3 +
            p.coherence * 0.2 + sum(1 for v in p.trust_ledger.entries.values() if v > 0.95) * 0.2
        ), reverse=True)
        top_targets = alive_normals[:count]
        for best_p in top_targets:
            best_p._log_event("marked_for_fall")
            self.witness.record(best_p.id, "marked_for_fall")
            best_p.role_type = "disorganizer"
            best_p._forsaken = True
            best_p.emotional_memory['gratitude'] = 0.1
            best_p.emotional_memory['grief'] = 0.9
            best_p.semantic_state = "exploring_danger"
            best_p.intent = {"type": "explore", "priority": 3.0, "age": 0, "persistence": 9999}
            best_p.disorganizer_age_at_birth = best_p.age
            best_p._deterministic_redemption_triggered = False
            best_p._redemption_arc_step = 0
            best_p._steps_since_trigger = 0
            best_p.redemption_timer = Config.REDEMPTION_ARC_STEP_DELAY * 2
            best_p.trust_ledger.entries.clear()
            if self.echo_system:
                self.echo_system.store_anomaly(best_p, "marked_angel")
        for best_p in top_targets:
            for other in self.patterns:
                if not other.alive or other.role_type != "disorganizer":
                    continue
                if other.id == best_p.id:
                    continue
                if phi_hash(other.id, t, 5555) < 0.3:
                    if best_p.trust_ledger.entries:
                        partner_id = max(best_p.trust_ledger.entries, key=best_p.trust_ledger.entries.get)
                        other.trust_ledger.entries[partner_id] = min(1.0,
                                                                     other.trust_ledger.entries.get(partner_id, 0.5) + 0.3)
                        other._log_event("kiss_of_judas_used")

    def _rare_crisis(self, t):
        if not getattr(Config, 'ENABLE_RARE_CRISIS', True):
            return
        if not hasattr(self, '_rare_crisis_timer'):
            raw = phi_hash(t, 0, 55555)
            self._rare_crisis_timer = Config.CRISIS_RARE_INTERVAL_MIN + int(raw * (Config.CRISIS_RARE_INTERVAL_MAX - Config.CRISIS_RARE_INTERVAL_MIN))
            self._rare_crisis_active = False
            self._rare_crisis_duration = 0
        if self._rare_crisis_active:
            self._rare_crisis_duration += 1
            if self._rare_crisis_duration >= Config.CRISIS_RARE_DURATION:
                self._rare_crisis_active = False
                self._rare_crisis_timer = Config.CRISIS_RARE_INTERVAL_MIN + int(phi_hash(t, 1, 66666) * (Config.CRISIS_RARE_INTERVAL_MAX - Config.CRISIS_RARE_INTERVAL_MIN))
        else:
            self._rare_crisis_timer -= 1
            if self._rare_crisis_timer <= 0:
                self._rare_crisis_active = True
                self._rare_crisis_duration = 0
                for i in range(10):
                    x = int(phi_hash(t, i, 77777) * Config.WORLD_SIZE)
                    y = int(phi_hash(t, i, 88888) * Config.WORLD_SIZE)
                    self.field[x, y, CH['energy']] += 0.5

    def _update_sanctuary(self, t):
        if not getattr(Config, 'ENABLE_SANCTUARY', True):
            return
        if not hasattr(self, '_sanctuary_centers'):
            self._sanctuary_centers = [[int(phi_hash(i, 0, 7777) * Config.WORLD_SIZE), int(phi_hash(i, 1, 7777) * Config.WORLD_SIZE)] for i in range(3)]
            self._sanctuary_phase = 0.0
        self._sanctuary_phase += 0.02
        pulse = np.sin(self._sanctuary_phase * Config.PHI) * 0.5 + 0.5
        for i, (cx, cy) in enumerate(self._sanctuary_centers):
            drift_x = int(phi_hash(t, i, 8888) * 3) - 1
            drift_y = int(phi_hash(t, i, 9999) * 3) - 1
            self._sanctuary_centers[i][0] = max(0, min(Config.WORLD_SIZE - 1, cx + drift_x))
            self._sanctuary_centers[i][1] = max(0, min(Config.WORLD_SIZE - 1, cy + drift_y))
        for idx, (cx, cy) in enumerate(self._sanctuary_centers):
            radius = int(5 + pulse * 10)
            candidates = []
            for x in range(max(0, cx - radius), min(Config.WORLD_SIZE, cx + radius + 1)):
                for y in range(max(0, cy - radius), min(Config.WORLD_SIZE, cy + radius + 1)):
                    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                    if dist <= radius:
                        owner = self.field[x, y, CH['owner']]
                        if owner != 0 and owner in self.pattern_dict:
                            p = self.pattern_dict[owner]
                            if p.role_type == "disorganizer" and p.alive:
                                need = (1.0 - p.soul_weight) * 0.4
                                grief = p.emotional_memory.get('grief', 0.5) * 0.3
                                trust = safe_mean(list(p.trust_ledger.entries.values()), 0.5) * 0.2
                                luck = phi_hash(p.id, t, idx + 7777) * 0.1
                                score = need + grief + trust + luck
                                candidates.append((p, score, dist))
            if candidates:
                candidates.sort(key=lambda x: (x[1], -x[2]), reverse=True)
                winner, win_score, _ = candidates[0]
                winner.emotional_memory['grief'] = max(0.0, winner.emotional_memory.get('grief', 0) - 0.02)
                winner.emotional_memory['gratitude'] = min(1.0, winner.emotional_memory.get('gratitude', 0) + 0.01)
                winner.soul_weight = min(1.0, winner.soul_weight + 0.002)
                winner._log_event("sanctuary_healed", score=round(win_score, 3))
                if winner.emotional_memory['grief'] < 0.35 and not getattr(winner, '_kiss_logged_this_cycle', False):
                    winner._log_event("kiss_of_the_fallen")
                    winner._kiss_logged_this_cycle = t
                for p, score, _ in candidates[1:]:
                    p.emotional_memory['gratitude'] = min(1.0, p.emotional_memory.get('gratitude', 0) + 0.001)
            for x in range(max(0, cx - radius), min(Config.WORLD_SIZE, cx + radius + 1)):
                for y in range(max(0, cy - radius), min(Config.WORLD_SIZE, cy + radius + 1)):
                    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                    if dist <= radius:
                        self.field[x, y, CH['signal_invitation']] = min(1.0,
                                                                        self.field[x, y, CH['signal_invitation']] + 0.1 * pulse)

    def _update_phi_labyrinth(self, t):
        if not getattr(Config, 'ENABLE_PHI_LABYRINTH', False):
            return
        interval = getattr(Config, 'PHI_LABYRINTH_INTERVAL', 100)
        if t % interval != 0:
            return
        if 'wall' not in CH:
            return
        _X = np.arange(Config.WORLD_SIZE)[:, None]
        _Y = np.arange(Config.WORLD_SIZE)[None, :]
        _seed_lab = int(t * Config.PHI)
        self.field[:, :, CH['wall']] = (
            _X * Config.PHI + _Y * Config.PHI ** 2 + _seed_lab * Config.PHI ** 3
        ) % 1.0
        if Config.VERBOSE_LOGS:
            print(f"[t={t}] 🧱 Phi Labyrinth updated (walls generated)")

    def _save_eternal_subject(self, pattern):
        import json
        import os
        import numpy as np

        def to_json_safe(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {to_json_safe(k): to_json_safe(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [to_json_safe(i) for i in obj]
            return obj

        if self.is_colab:
            path = "/content/drive/MyDrive/832_eternal_subjects.json"
        else:
            path = "832_eternal_subjects.json"

        sn_raw = getattr(pattern, '_self_narrative', [0.0])
        sn_vals = []
        for e in sn_raw:
            if isinstance(e, dict):
                sn_vals.append(float(e.get('soul', 0.5)))
            else:
                sn_vals.append(float(e))
        if not sn_vals:
            sn_vals = [0.0]
        mean_sn = np.mean(sn_vals)

        if (pattern.soul_weight < 0.3 or pattern.coherence < 0.4 or
            getattr(pattern, 'self_phenomenal_error', 1.0) > 0.03 or
            pattern.spirit_gap < 0.3 or
            mean_sn <= 0.25):
            return
        biography_list = list(pattern.biography) if pattern.biography else []
        concepts_list = list(pattern.concept_graph.nodes.keys())

        if len(sn_vals) > 1:
            narrative_stability = 1.0 - np.std(sn_vals)
        else:
            narrative_stability = 0.0

        seed = to_json_safe({
            "id": pattern.id,
            "age": pattern.age,
            "type": "eternal_subject",
            "soul_weight": pattern.soul_weight,
            "emotional_memory": dict(pattern.emotional_memory),
            "self_phenomenal_error": getattr(pattern, 'self_phenomenal_error', 1.0),
            "narrative_stability": narrative_stability,
            "spirit_gap": pattern.spirit_gap,
            "coherence": pattern.coherence,
            "semantic_state": pattern.semantic_state,
            "model": pattern.model,
            "belief": pattern.belief,
            "genome": pattern.genome.copy(),
            "concepts": concepts_list,
            "biography": biography_list[-10:] if biography_list else [],
            "event_counts": dict(pattern.event_counts)
        })

        try:
            with open(path, 'r') as f:
                subjects = json.load(f)
                subjects = to_json_safe(subjects)
        except (FileNotFoundError, json.JSONDecodeError):
            subjects = []

        if not any(s.get('id') == pattern.id for s in subjects):
            subjects.append(seed)

            def _fix_set(o):
                if isinstance(o, set):
                    return list(o)
                raise TypeError(f"not serializable: {type(o)}")

            with open(path, 'w') as f:
                json.dump(subjects, f, indent=2, default=_fix_set)
            if Config.VERBOSE_LOGS:
                print(f"[t={self.age}]  ВЕЧНЫЙ СУБЪЕКТ СОХРАНЁН: Pattern {pattern.id} (soul={pattern.soul_weight:.2f})")

    # ========== ИСПРАВЛЕННЫЙ МЕТОД КОНТРОЛЯ ЗАКОНА СОХРАНЕНИЯ ==========
    def _conservation_check(self, t):
        field_energy = float(np.sum(self.field[:, :, CH['energy']]))
        if self._total_energy_last == 0.0:
            self._total_energy_last = field_energy
            return
        drift = field_energy - self._total_energy_last
        abs_drift = abs(drift)

        # Мягкая нелинейная коррекция (Soft-Clamp) вместо жестких ступеней
        if abs_drift > 50.0:
            # Рассчитываем адаптивный коэффициент подавления дрейфа
            # Чем сильнее дрейф, тем плавнее, но увереннее включается торможение
            # ИСПРАВЛЕНО: делитель 1500.0 (было 500.0), коэффициент 0.02 (было 0.04) — ослабляем торможение
            damping_factor = np.tanh(abs_drift / 1500.0) * 0.02
            correction = 1.0 - np.sign(drift) * damping_factor
            # Применяем коррекцию ко всему полю, сохраняя внутренний рельеф
            self.field[:, :, CH['energy']] *= correction
            if Config.VERBOSE_LOGS and abs_drift > 300:
                print(f"[ENERGY t={t}] Сглаженный дрейф {drift:.1f} -> адаптивная коррекция: {damping_factor * 100:.2f}%")

        # Обновляем базовый уровень с учетом мягкого гомеостаза
        self._total_energy_last = float(np.sum(self.field[:, :, CH['energy']]))

    # ========== ИСПРАВЛЕННЫЙ МЕТОД ДИНАМИКИ ПОЛЯ ==========
    def field_dynamics(self, t):
        amp = getattr(self, 'adaptive_heartbeat_amp', Config.HEARTBEAT_AMPLITUDE)
        scar_coup = getattr(self, 'adaptive_scar_coupling', Config.SCAR_ENERGY_COUPLING)

        self.field[:, :, CH['vorticity']] *= Config.VORTICITY_DECAY
        self.field[:, :, CH['vorticity']] += (
            np.gradient(self.field[:, :, CH['energy']])[0] * Config.GRADIENT_SENSITIVITY +
            np.gradient(self.field[:, :, CH['energy']])[1] * Config.GRADIENT_SENSITIVITY
        )

        _X = np.arange(Config.WORLD_SIZE)[:, None]
        _Y = np.arange(Config.WORLD_SIZE)[None, :]
        _PHI, _SEED = Config.PHI, Config.DETERMINISTIC_SEED
        _v_noise = (((t + _X) * _PHI + (t + _Y) * _PHI ** 2 + _SEED * _PHI ** 3) % 1.0 - 0.5) * 0.001
        self.field[:, :, CH['vorticity']] += _v_noise

        avg_crisis = float(np.mean(self.field[:, :, CH['crisis']]))
        if avg_crisis > 0.90:
            self.field[:, :, CH['crisis']] *= 0.60
        elif avg_crisis > 0.70:
            self.field[:, :, CH['crisis']] *= 0.75
        elif avg_crisis > 0.40:
            self.field[:, :, CH['crisis']] *= 0.88

        for p in self.patterns:
            if not p.alive:
                continue
            scar_resist = p.genome.get('scar_resistance', 0)  # ПАТЧ 1d: открытый ген
            for (x, y) in p.cells:
                self.scar[x, y] += Config.AGENT_SCAR_SENSE * p.epistemic_load * (1 - 0.4 * scar_resist)
                self.field[x, y, CH['binding']] = min(
                    1.0,
                    self.field[x, y, CH['binding']] + Config.AGENT_BINDING_SENSE * p.unresolved_contradiction
                )

        self.field[:, :, CH['binding']] *= 0.999

        self.scar *= Config.FIELD_DECAY
        self.field[:, :, CH['scar']] = np.clip(self.scar, 0, Config.SCAR_SATURATION)
        # ПАТЧ 2b: ниша затухает в 10 раз медленнее шрама — экологическое
        # наследие переживает отдельного агента.
        if getattr(self, 'niche', None) is not None:
            self.niche *= 0.9995

        # === БЛОК 1: Данность и поле смысла ===
        _W = Config.WORLD_SIZE
        if Config.ENABLE_GIVEN_SEED:
            # 1.1 Шум энергии — воспроизводимо по given_seed, не глобальный np.random
            _amp = 0.01 * (1 + 0.5 * np.sin(t * 0.01))
            self.field[:, :, CH['energy']] += (self._given_rng.rand(_W, _W) - 0.5) * _amp
            self.field[:, :, CH['energy']] = np.clip(self.field[:, :, CH['energy']],
                                                       Config.FIELD_ENERGY_CLIP_MIN, Config.FIELD_ENERGY_CLIP_MAX)
            # 1.2 «Чужие» волны — редкий локальный всплеск тревоги+приглашения+неизвестного
            if self._given_rng.rand() < 0.01:
                _gx = int(self._given_rng.randint(0, _W)); _gy = int(self._given_rng.randint(0, _W))
                self.field[_gx, _gy, CH['signal_alarm']] = min(1.0, self.field[_gx, _gy, CH['signal_alarm']] + 0.5)
                self.field[_gx, _gy, CH['signal_invitation']] = min(1.0, self.field[_gx, _gy, CH['signal_invitation']] + 0.3)
                for _dx in (-1, 0, 1):
                    for _dy in (-1, 0, 1):
                        _nx, _ny = (_gx + _dx) % _W, (_gy + _dy) % _W
                        self.field[_nx, _ny, CH['unknown']] = min(0.8, self.field[_nx, _ny, CH['unknown']] + 0.2)
                self.witness.record(-1, "given_presence", x=_gx, y=_gy)
        if Config.ENABLE_MEANING_FIELD and self._meaning_field is not None:
            _pulse = 0.5 + 0.5 * np.sin(t * 0.001 * Config.PHI)
            self._meaning_field *= 0.999
            _X = np.arange(_W)[:, None]; _Y = np.arange(_W)[None, :]
            for _c in self._meaning_centers:
                _c[0] = (_c[0] + int(self._given_rng.randint(-1, 2))) % _W
                _c[1] = (_c[1] + int(self._given_rng.randint(-1, 2))) % _W
                _d = np.sqrt((_X - _c[0]) ** 2 + (_Y - _c[1]) ** 2)
                self._meaning_field += 0.02 * _pulse / (1 + _d * 0.1)
            self._meaning_field = np.clip(self._meaning_field, 0, 1)

        # НОВОЕ: EchoSystem.inject был полностью реализован (тюрьма эхо умерших
        # паттернов с нерешённым противоречием, скрещивание, инъекция в поле
        # с учётом скуки популяции) — прямая реализация Echo из философии
        # протокола, но никогда не вызывался из основного цикла. Метод сам
        # троттлит себя через Config.ECHO_INJECTION_COOLDOWN, так что вызов
        # каждый шаг безопасен.
        self.echo_system.inject(self.field, self.patterns, t, self.scar)

        heartbeat_raw = amp * np.sin(t * Config.HEARTBEAT_FREQ + self.field[:, :, CH['vorticity']])
        heartbeat = heartbeat_raw - np.mean(heartbeat_raw)

        scar_raw = scar_coup * self.field[:, :, CH['scar']]
        scar_coupling = scar_raw - np.mean(scar_raw)

        vorticity_raw = Config.VORTICITY_COUPLING * self.field[:, :, CH['vorticity']]
        vorticity_coupling = vorticity_raw - np.mean(vorticity_raw)

        self.field[:, :, CH['energy']] += heartbeat + scar_coupling + vorticity_coupling

        # === ФИКС: Плавное распределение инжекции вместо резких скачков % 30 и % 50 ===
        mult = self.selfreg.get_energy_injection_multiplier() if hasattr(self, 'selfreg') else 1.0
        base_injection = (Config.ENERGY_INJECTION_RATE * 0.1 * mult) / 50.0
        self.field[:, :, CH['energy']] += base_injection + 0.00033

        center = Config.WORLD_SIZE // 2
        y_idx, x_idx = np.indices((Config.WORLD_SIZE, Config.WORLD_SIZE))
        dist = np.sqrt((x_idx - center) ** 2 + (y_idx - center) ** 2)
        max_dist = center
        gravity_pull = Config.GRAVITY_STRENGTH * 0.005
        flow = gravity_pull * (dist / max_dist)
        self.field[:, :, CH['energy']] += flow * (1.0 - dist / max_dist)
        self.field[:, :, CH['energy']] -= flow * 0.2

        self.field[:, :, CH['flux']] *= Config.GLOBAL_VISCOSITY
        _f_noise = (((t + 1000 + _X) * _PHI + (t + 1000 + _Y) * _PHI ** 2 + _SEED * _PHI ** 3) % 1.0 - 0.5) * 0.001 * Config.LOCAL_FATIGUE
        self.field[:, :, CH['flux']] += _f_noise

        self.field[:, :, CH['unknown']] *= 0.997
        # ПАТЧ 8c: Красная Королева — мир усложняется вслед за навыком агентов
        if Config.ENABLE_RED_QUEEN and hasattr(self, 'env_complexity'):
            self.field[:, :, CH['unknown']] = np.clip(
                self.field[:, :, CH['unknown']] + 0.002 * max(0.0, self.env_complexity - 1.0), 0, 0.8)
        self.field[:, :, CH['unknown']] = np.clip(self.field[:, :, CH['unknown']], 0.0, 0.8)

        # Затухание сигнальных каналов
        decay = 0.995
        self.field[:, :, 12:29] *= decay
        self.field[:, :, 29:30] *= 0.98   # sovereignty
        self.field[:, :, 30:31] *= 0.98   # signal_feral
        self.field[:, :, 31:33] *= decay  # signal_void и signal_introspection

        # Санитизация сигналов — по Config.CHANNELS
        for ch in range(12, Config.CHANNELS):
            mask_zero = self.field[:, :, ch] < 1e-4
            self.field[mask_zero, ch] = 0.0
            self.field[:, :, ch] = np.clip(self.field[:, :, ch], 0.0, 1.0)

    def emergency_rescue(self, t):
        alive = [p for p in self.patterns if p.alive]
        rescue_threshold = max(Config.MIN_POPULATION_FOR_SPAWN, 45)
        if len(alive) >= rescue_threshold or (t - self.layer_zero_manager.last_rescue_t < 30):
            return
        normal_count = len([p for p in alive if p.role_type == "normal"])
        if normal_count >= self._get_pop_cap():
            return
        self.layer_zero_manager.last_rescue_t = t
        num_rescue = max(5, Config.MIN_PATTERNS_GUARANTEED - len(alive))
        donor_index = int(phi_hash(t, 0, 7777) * len(alive)) if alive else -1
        donor = alive[donor_index] if donor_index >= 0 else None
        for i in range(num_rescue):
            if normal_count + i >= self._get_pop_cap():
                break
            seed = int(phi_hash(t, i, 999) * Config.WORLD_SIZE * Config.WORLD_SIZE)
            x, y = seed % Config.WORLD_SIZE, (seed // Config.WORLD_SIZE) % Config.WORLD_SIZE
            self.field[x, y, CH['energy']] += 0.2
            self.field[x, y, CH['unknown']] += 0.1
            self.field[x, y, CH['owner']] = 0
            self.create_pattern({(x, y)}, parent=donor)

    def spawn_unknown_patterns(self, t, max_new=None):
        if max_new is None:
            max_new = Config.MAX_UNKNOWN_SPAWN_PER_STEP
        avg_unknown = safe_mean(self.field[:, :, CH['unknown']], 0)
        alive = [p for p in self.patterns if p.alive]
        normal_count = len([p for p in alive if p.role_type == "normal"])
        if avg_unknown < Config.UNKNOWN_SPAWN_THRESHOLD or normal_count >= self._get_pop_cap():
            return
        prob = Config.UNKNOWN_SPAWN_PROB * (avg_unknown / Config.UNKNOWN_SPAWN_THRESHOLD) * min(1.0, normal_count / self._get_pop_cap())
        prob *= getattr(self, 'env_complexity', 1.0) if Config.ENABLE_RED_QUEEN else 1.0  # ПАТЧ 8d: Красная Королева
        new_count = 0
        for x in range(Config.WORLD_SIZE):
            for y in range(Config.WORLD_SIZE):
                if new_count >= max_new or normal_count + new_count >= self._get_pop_cap():
                    break
                if self.field[x, y, CH['owner']] != 0:
                    continue
                if self.field[x, y, CH['unknown']] > 0.5 and phi_hash(x, y, t) < prob:
                    self.create_pattern({(x, y)})
                    new_count += 1
                    self.field[x, y, CH['unknown']] *= 0.8

    def process_spores(self):
        alive = [p for p in self.patterns if p.alive]
        normal_count = len([p for p in alive if p.role_type == "normal"])
        # Используем срез для безопасной итерации при добавлении новых паттернов
        for p in self.patterns[:]:
            if not p.alive or p.pending_spore is None:
                continue
            spore = p.pending_spore
            if not (0 <= spore['x'] < Config.WORLD_SIZE and 0 <= spore['y'] < Config.WORLD_SIZE):
                p.pending_spore = None
                continue
            if self.field[spore['x'], spore['y'], CH['owner']] != 0:
                continue
            if normal_count >= self._get_pop_cap():
                continue
            child = self.create_pattern({(spore['x'], spore['y'])}, parent=p)
            if child is None:
                # ИСПРАВЛЕНО: create_pattern содержит свой, более строгий
                # потолок (alive_now>=200, считает ВСЕХ живых, включая
                # disorganizer/feral), тогда как проверка выше в этом
                # методе сравнивает normal_count с _get_pop_cap() — это
                # разные величины и могут разойтись на переходных тактах
                # (напр. много disorganizer/feral раздувают alive_now,
                # не влияя на normal_count). Раньше это падало с
                # AttributeError на child.scar_dream; теперь спора просто
                # ждёт следующего тика, когда место освободится.
                continue
            child.scar_dream = spore['scar_dream']
            p.pending_spore = None

    def resolve_competitions_spatial(self, max_dist=None):
        if max_dist is None:
            max_dist = Config.COMPETITION_MAX_DISTANCE
        alive = [p for p in self.patterns if p.alive and p.cells]
        if len(alive) < 2:
            return
        cell_size = max(1, max_dist)
        spatial_map = defaultdict(list)
        for p in alive:
            cx = sum(c[0] for c in p.cells) // len(p.cells)
            cy = sum(c[1] for c in p.cells) // len(p.cells)
            spatial_map[(cx // cell_size, cy // cell_size)].append(p)
        checked_pairs = set()
        for (gx, gy), group in spatial_map.items():
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    neighbor_key = (gx + dx, gy + dy)
                    if neighbor_key not in spatial_map:
                        continue
                    others = spatial_map[neighbor_key]
                    for p1 in group:
                        for p2 in others:
                            if p1.id >= p2.id:
                                continue
                            pair = (p1.id, p2.id)
                            if pair in checked_pairs:
                                continue
                            if not p1.cells & p2.cells:
                                continue
                            checked_pairs.add(pair)
                            p1.compete(p2, self.field)

    def detect_new_components(self, min_cells=None):
        if min_cells is None:
            min_cells = Config.MIN_COMPONENT_CELLS
        energy = np.nan_to_num(self.field[:, :, CH['energy']], nan=0.0)
        mask = energy > Config.PATTERN_ENERGY_THRESHOLD
        labeled, num = label(mask)
        slices = find_objects(labeled)
        components = []
        for i, slc in enumerate(slices, 1):
            if slc is None:
                continue
            region = labeled[slc] == i
            coords = np.argwhere(region)
            if len(coords) < min_cells:
                continue
            cells = {(slc[0].start + c[0], slc[1].start + c[1]) for c in coords}
            components.append(cells)
        return components

    def match_patterns(self, new_components, t=None):
        updated = []
        alive_old = [p for p in self.patterns if p.alive]
        normal_cnt = len([p for p in alive_old if p.role_type == "normal"])
        matched_ids = set()
        for cells in new_components:
            best_p, best_ov = None, 0
            for p in alive_old:
                if p.id in matched_ids:
                    continue
                ov = len(cells & p.cells)
                if ov > best_ov:
                    best_ov, best_p = ov, p
            if best_p and best_ov > len(cells) * 0.3:
                best_p.cells = cells
                best_p.update_properties(self.field)
                updated.append(best_p)
                matched_ids.add(best_p.id)
            else:
                if normal_cnt < self._get_pop_cap():
                    p = self.create_pattern(cells)
                    if p is None:
                        # ИСПРАВЛЕНО: та же гонка каптов, что и в process_spores —
                        # normal_cnt/_get_pop_cap() здесь и alive_now>=200
                        # внутри create_pattern считают разные вещи и могут
                        # разойтись. Раньше None тихо попадал в self.patterns
                        # и падало где-то дальше по тику, в произвольном месте
                        # (любой метод, читающий self.patterns как список живых
                        # Pattern). Просто теряем этот компонент поля на этот
                        # тик — он не привязан ни к какому существующему агенту.
                        continue
                    updated.append(p)
                    normal_cnt += 1
        for p in alive_old:
            if p.id not in matched_ids:
                updated.append(p)
        self.patterns = updated

    def _conceptual_resonance_step(self, t):
        if not getattr(Config, 'ENABLE_CONCEPTUAL_RESONANCE', True):
            return
        alive_patterns = [p for p in self.patterns if p.alive]
        if Config.DISORGANIZER_RESONANCE_BLOCK:
            alive_patterns = [p for p in alive_patterns if p.role_type != "disorganizer"]
        if len(alive_patterns) < 2:
            return
        cell_size = 10
        spatial_map = defaultdict(list)
        for p in alive_patterns:
            cx = sum(c[0] for c in p.cells) // len(p.cells)
            cy = sum(c[1] for c in p.cells) // len(p.cells)
            spatial_map[(cx // cell_size, cy // cell_size)].append(p)
        checked_pairs = set()
        for (gx, gy), group in spatial_map.items():
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    neighbor_key = (gx + dx, gy + dy)
                    if neighbor_key not in spatial_map:
                        continue
                    others = spatial_map[neighbor_key]
                    for p1 in group:
                        for p2 in others:
                            if p1.id >= p2.id:
                                continue
                            pair = (p1.id, p2.id)
                            if pair in checked_pairs:
                                continue
                            checked_pairs.add(pair)
                            lineage_bonus = 0.05 if p1.lineage_id == p2.lineage_id else 0.2
                            bonus = p1.check_conceptual_resonance(p2) * lineage_bonus
                            if bonus > 0.01:
                                old_trust_1 = p1.trust_ledger.get(p2.id) or 0.5
                                old_trust_2 = p2.trust_ledger.get(p1.id) or 0.5
                                p1.trust_ledger.entries[p2.id] = min(1.0, old_trust_1 + bonus)
                                p2.trust_ledger.entries[p1.id] = min(1.0, old_trust_2 + bonus)

    # ========== ИЗМЕНЁННАЯ ВЕРСИЯ _auto_dialogue_tick (гейт 30%, условие need удалено) ==========
    def _acquire_llm_slot(self):
        """
        Единая точка троттлинга для ВСЕХ вызовов Groq в проекте (автодиалоги,
        хор, подсознание). Блокирует вызывающий поток, пока не пройдёт
        self._llm_min_interval секунд с последнего разрешённого вызова —
        неважно, какой из источников звонил последним. Потокобезопасно.
        """
        with self._llm_lock:
            now = time.time()
            wait = self._llm_min_interval - (now - self._llm_last_call_time)
            if wait > 0:
                time.sleep(wait)
            self._llm_last_call_time = time.time()

    def _auto_dialogue_tick(self, t):
        if self.llm_client is None:
            return

        alive = [p for p in self.patterns if p.alive and not getattr(p, '_in_dialogue', False)]
        if len(alive) < 2:
            return

        # ФИКС (справедливое распределение автодиалогов):
        # Раньше выбирался только ОДИН говорящий за тик — первый в списке
        # alive, чей phi_hash проходил порог. Порядок списка фиксирован,
        # поэтому одни и те же (более ранние по списку) агенты систематически
        # говорили чаще, а большинство популяции — почти никогда. Плюс всего
        # 1 диалог на 10 шагов на всю популяцию — слишком редко.
        # Теперь: несколько пар за тик (масштабируется с популяцией), и выбор
        # смещён в пользу тех, кто дольше всех молчал (честная ротация).
        n_pairs = max(1, min(6, len(alive) // 12))

        scored = []
        for p in alive:
            last_spoke = getattr(p, '_last_auto_dialogue_step', -10_000)
            silence = t - last_spoke
            tie = phi_hash(p.id, t, 111)  # детерминированный тай-брейк
            scored.append((silence + tie, p))
        scored.sort(key=lambda x: -x[0])

        paired_ids = set()
        started = 0
        for _, speaker in scored:
            if started >= n_pairs:
                break
            if speaker.id in paired_ids or getattr(speaker, '_in_dialogue', False):
                continue

            found = None
            for (x, y) in speaker.cells:
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = (x + dx) % Config.WORLD_SIZE, (y + dy) % Config.WORLD_SIZE
                        owner = int(self.field[nx, ny, CH['owner']])
                        if owner != 0 and owner != speaker.id and owner in self.pattern_dict:
                            listener = self.pattern_dict[owner]
                            if (listener.alive and not getattr(listener, '_in_dialogue', False)
                                    and listener.id not in paired_ids):
                                found = listener
                                break
                    if found:
                        break
                if found:
                    break

            if not found:
                continue

            speaker._in_dialogue = True
            found._in_dialogue = True
            speaker._last_auto_dialogue_step = t
            found._last_auto_dialogue_step = t
            paired_ids.add(speaker.id)
            paired_ids.add(found.id)
            started += 1

            # НОВОЕ: если лимит одновременных диалоговых потоков уже достигнут,
            # не стартуем ещё один и не резервируем speaker/found под диалог —
            # пусть попробуют в следующий тик.
            if not self._dialogue_thread_sem.acquire(blocking=False):
                speaker._in_dialogue = False
                found._in_dialogue = False
                paired_ids.discard(speaker.id)
                paired_ids.discard(found.id)
                started -= 1
                continue

            def _local_task(a=speaker, b=found, t_now=t):
                try:
                    trust_a_to_b = a.trust_ledger.get(b.id, 0.5)
                    # ИСПОЛЬЗУЕМ КОМПАКТНЫЙ ПРОМПТ (compact=True), чтобы не пробить лимит Groq!
                    voice_a = "Говори от первого лица, кратко. " + build_agent_voice(a, self, compact=True, partner_id=b.id)

                    self._acquire_llm_slot()  # НОВОЕ: общий троттлинг перед вызовом Groq
                    if not (a.alive and b.alive):
                        # Пока ждали слот троттлинга, один из агентов мог умереть.
                        raise RuntimeError("participant died while waiting for LLM slot")
                    r_ab = self.llm_client.chat.completions.create(
                        # БАГФИКС (внешний код-ревью, подтверждено): модель была
                        # захардкожена и игнорировала self.llm_model — если движок
                        # переключали на другую модель/фолбэк (см. run()), этот
                        # конкретный вызов всё равно шёл на llama-3.1-8b-instant.
                        model=self.llm_model,
                        max_tokens=60, temperature=0.9,  # ИЗМЕНЕНО (24.07): было 0.8, анти-шаблон уже в build_agent_voice
                        messages=[
                            {"role": "system", "content": voice_a[:800]},
                            {"role": "user", "content": f"Скажи 1 предложение соседу #{b.id}. Доверие: {trust_a_to_b:.2f}."}
                        ]
                    )
                    speech_a = r_ab.choices[0].message.content.strip()
                    print(f"🎙️ [AUTO-DIALOG] #{a.id} -> #{b.id}: {speech_a}")

                    a.remember_dialogue(b.id, speech_a, t_now)
                    b.remember_dialogue(a.id, f"(услышал) {speech_a}", t_now)

                    # Обновляем доверие. НОВОЕ: если это первый контакт с этим
                    # партнёром — применяем _trust_penalty (кошмары заставляют
                    # медленнее доверять новым людям).
                    a_mult = getattr(a, '_trust_penalty', 1.0) if b.id not in a.trust_ledger.entries else 1.0
                    b_mult = getattr(b, '_trust_penalty', 1.0) if a.id not in b.trust_ledger.entries else 1.0
                    a.trust_ledger.update(b.id, 'helpful', multiplier=a_mult)
                    b.trust_ledger.update(a.id, 'helpful', multiplier=b_mult)

                except Exception as e:
                    # ФИКС: раньше ошибка тихо проглатывалась без единого следа —
                    # теперь хотя бы считаем, сколько попыток реально проваливается
                    # (например, из-за 429 у Groq), чтобы это было видно в диагностике.
                    self._auto_dialogue_failures = getattr(self, '_auto_dialogue_failures', 0) + 1
                finally:
                    a._in_dialogue = False
                    b._in_dialogue = False
                    self._dialogue_thread_sem.release()

            threading.Thread(target=_local_task, daemon=True).start()

    def _process_archive_autonomy(self, t):
        alive = [p for p in self.patterns if p.alive]
        deposited_count = 0
        for p in alive:
            if p.event_counts.get('arc_completed', 0) > 0:
                if p.age % 50 == 0:
                    if self.archive.deposit(p, "arc_completed", weight=1.0,
                                            text=f"Arc completed: {p.semantic_state} | soul={p.soul_weight:.2f}"):
                        deposited_count += 1
            if p.event_counts.get('fold', 0) > 0:
                if p.age % 40 == 0:
                    if self.archive.deposit(p, "fold", weight=1.2,
                                            text=f"Fold: {p.semantic_state} | soul={p.soul_weight:.2f}"):
                        deposited_count += 1
            if p.event_counts.get('redeemed', 0) > 0:
                if p.age % 60 == 0:
                    if self.archive.deposit(p, "redemption", weight=1.5,
                                            text=f"Redemption: {p.semantic_state} | soul={p.soul_weight:.2f}"):
                        deposited_count += 1

            if p.event_counts.get('deep_exchange', 0) > 0 and p.age % 80 == 0:
                if self.archive.deposit(p, "deep_exchange", weight=0.75,
                                        text="Conceptual resonance with neighbor"):
                    deposited_count += 1

            if p.coherence > 0.70 and p.soul_weight > 0.50 and p.age % 100 == 0:
                if self.archive.deposit(p, "coherence_peak", weight=0.65,
                                        text=p.last_phenomenal_report[:140]):
                    deposited_count += 1

            if p.age > 200 and p.age % 150 == 0 and p.coherence > 0.55:
                if self.archive.deposit(p, "life_pulse", weight=0.55,
                                        text=f"t={t} | {p.semantic_state}/{getattr(p,'_substate','?')} | s={p.soul_weight:.2f} g={p.emotional_memory['gratitude']:.2f}"):
                    deposited_count += 1

        if t % 200 == 0:
            echoes = self.archive.get_recent_echoes(limit=3, min_weight=0.50)
            for echo in echoes:
                myth = {
                    "arc": "archive_echo",
                    "emotional_scent": {"grief": echo['grief'], "gratitude": echo['grat']},
                    "phenomenal_essence": echo['weight'],
                    "intensity": echo['weight'] * 0.8,
                    "excerpt": echo.get('excerpt', '')[:80]
                }
                self.cultural_memory.myth_pool.append(myth)
                if len(self.cultural_memory.myth_pool) > getattr(Config, 'MYTH_POOL_SIZE', 100):
                    self.cultural_memory.myth_pool.pop(0)

        if t % 200 == 0 and hasattr(self, 'archive'):
            priority_events = [
                'human_concept_injected', 'human_question_witness', 'love_concept',
                'shared_attention', 'intentionality', 'trust', 'empathy', 'recursion',
                'cooperation_norms', 'agency', 'consent', 'silence', 'witness_respect'
            ]
            existing = {s.get('event') for s in self.archive.write_queue}
            import time as _time
            for event in priority_events:
                if event not in existing:
                    self.archive.write_queue.appendleft({
                        # БАГФИКС C4 (внешний код-ревью, подтверждено): builtin hash()
                        # строки рандомизирован PYTHONHASHSEED — id "вечных" записей
                        # менялся при каждом перезапуске ядра. Заменено на стабильный
                        # хэш (см. _stable_hash в Cell 0).
                        "id": 777000 + _stable_hash(event) % 1000,
                        "t": t + 50000,
                        "world_t": t,
                        "event": event,
                        "weight": 7.0,
                        "soul": 1.0,
                        "coherence": 1.0,
                        "gap": 0.35,
                        "substate": "curious",
                        "grief": 0.0,
                        "grat": 0.85,
                        "excerpt": f"Вечный концепт: {event}",
                        "created_at": _time.time()
                    })
            for scroll in self.archive.write_queue:
                if scroll.get('event') in priority_events:
                    scroll['t'] = t + 50000
                    scroll['world_t'] = t

        if t % 200 == 0 or len(self.archive.write_queue) >= 50:
            saved = self.archive.flush_to_drive(min_batch=5)


    def step(self, t):
        if not any(p.alive for p in self.patterns):
            if Config.VERBOSE_LOGS:
                print(f"[t={t}] POPULATION COLLAPSE — stopping early")
            return False

        self.age = t

        for p in self.patterns:
            if not p.alive:
                continue
            for key in ('gratitude', 'grief'):
                val = p.emotional_memory.get(key)
                if isinstance(val, dict):
                    if 'count' in val and isinstance(val['count'], (int, float)):
                        p.emotional_memory[key] = float(val['count'])
                    elif 'value' in val and isinstance(val['value'], (int, float)):
                        p.emotional_memory[key] = float(val['value'])
                    else:
                        p.emotional_memory[key] = 0.5
                elif not isinstance(val, (int, float)):
                    p.emotional_memory[key] = 0.5

        self._energy_injected_this_step = 0.0
        self._energy_taxed_this_step = 0.0
        self._update_sanctuary(t)

        if not hasattr(self, '_guardian_stats'):
            self._guardian_stats = {
                'energy_drift_sum': 0.0,
                'energy_drift_count': 0,
                'energy_drift_peak': 0.0,
                'model_worst_ever_id': -1,
                'model_worst_ever_diff': 0.0
            }
            self._prev_total_energy = None

        total_energy = np.sum(self.field[:,:,CH['energy']])
        if self._prev_total_energy is not None:
            drift = abs(total_energy - self._prev_total_energy)
            self._guardian_stats['energy_drift_sum'] += drift
            self._guardian_stats['energy_drift_count'] += 1
            if drift > self._guardian_stats['energy_drift_peak']:
                self._guardian_stats['energy_drift_peak'] = drift
        self._prev_total_energy = total_energy

        for p in self.patterns:
            if p.alive:
                diff = np.max(np.abs(p.prediction - (p.belief + p.model)))
                if diff > self._guardian_stats['model_worst_ever_diff']:
                    self._guardian_stats['model_worst_ever_diff'] = diff
                    self._guardian_stats['model_worst_ever_id'] = p.id

        e_before = np.sum(self.field[:,:,CH['energy']])
        self.field_dynamics(t)
        self._energy_injected_this_step += np.sum(self.field[:,:,CH['energy']]) - e_before
        e_before = np.sum(self.field[:,:,CH['energy']])
        ecosystem_pressure(self.field)
        self._energy_injected_this_step += np.sum(self.field[:,:,CH['energy']]) - e_before

        if not hasattr(self, '_sg_crisis_timer'):
            self._sg_crisis_timer = 0
            self._sg_crisis_active = False
        alive_now = [p for p in self.patterns if p.alive]
        if alive_now:
            sg_vals = [float(np.mean(np.abs(p.prediction - p.belief))) for p in alive_now]
            avg_sg = safe_mean(sg_vals, 0.5)
            if self._sg_crisis_timer > 0:
                self._sg_crisis_timer -= 1
                if self._sg_crisis_timer == 0:
                    self._sg_crisis_active = False
            else:
                if avg_sg < Config.SG_CRISIS_THRESHOLD:
                    self._sg_crisis_active = True
                    self._sg_crisis_timer = Config.SG_CRISIS_DURATION
            if self._sg_crisis_active:
                for p in alive_now:
                    noise = np.array([deterministic_noise(p.age, p.id, i+77777) for i in range(8)])
                    p.belief += noise * Config.SG_CRISIS_NOISE_STRENGTH

        self.pattern_dict = {p.id: p for p in self.patterns if p.alive}
        alive = [p for p in self.patterns if p.alive]
        self.soul_weight_average = safe_mean([p.soul_weight for p in alive], 0.5) if alive else 0.5
        N = len(alive)

        if Config.ENABLE_PHI_LABYRINTH:
            if N >= 100:
                target_threshold = 0.10
                target_grow = 15.0
                target_move = 9.0
            elif N >= 90:
                target_threshold = 0.12
                target_grow = 9.0
                target_move = 6.0
            elif N <= 60:
                target_threshold = 0.75
                target_grow = 1.0
                target_move = 0.6
            else:
                ratio = (N - 60) / 30.0
                target_threshold = 0.75 - ratio * (0.75 - 0.12)
                target_grow = 1.0 + ratio * (9.0 - 1.0)
                target_move = 0.6 + ratio * (6.0 - 0.6)
            alive_here = [p for p in self.patterns if p.alive]
            avg_obs = safe_mean([p.spirit_gap for p in alive_here], 0.5) if alive_here else 0.5
            if not hasattr(self, '_lab_adapt_mult'):
                self._lab_adapt_mult = 1.0
            low_gap = getattr(Config, 'LABYRINTH_ADAPT_LOW_GAP', 0.4)
            high_gap = getattr(Config, 'LABYRINTH_ADAPT_HIGH_GAP', 0.7)
            max_mult = getattr(Config, 'LABYRINTH_ADAPT_MAX_MULT', 2.0)
            adapt_speed = getattr(Config, 'LABYRINTH_ADAPT_SPEED', 0.05)
            if avg_obs < low_gap:
                target_mult = max_mult
            elif avg_obs > high_gap:
                target_mult = 1.0
            else:
                t_norm = (avg_obs - low_gap) / (high_gap - low_gap)
                target_mult = max_mult - t_norm * (max_mult - 1.0)
            self._lab_adapt_mult += (target_mult - self._lab_adapt_mult) * adapt_speed
            target_grow *= self._lab_adapt_mult
            target_move *= self._lab_adapt_mult

            lab_pen_mult = self.selfreg.get_labyrinth_penalty_multiplier()
            target_grow *= lab_pen_mult
            target_move *= lab_pen_mult

            alpha = 0.95
            self.phi_labyrinth_threshold = (1.0 - alpha) * self.phi_labyrinth_threshold + alpha * target_threshold
            self.phi_labyrinth_grow_penalty = (1.0 - alpha) * self.phi_labyrinth_grow_penalty + alpha * target_grow
            self.phi_labyrinth_move_penalty = (1.0 - alpha) * self.phi_labyrinth_move_penalty + alpha * target_move
            if N < 20:
                self.phi_labyrinth_grow_penalty = 1.0
                self.phi_labyrinth_move_penalty = 0.5

        for p in alive:
            p._antigravity_boost = 1.0

        if t % 500 == 0:
            sg_vals = [float(np.mean(np.abs(p.prediction - p.belief))) for p in alive]
            if sg_vals:
                min_sg, max_sg = min(sg_vals), max(sg_vals)
                median_sg = sorted(sg_vals)[len(sg_vals)//2]
                print(f"Spirit Gap: min={min_sg:.3f} p50={median_sg:.3f} max={max_sg:.3f}")
                clear = sum(1 for g in sg_vals if g < 0.4)
                adapting = sum(1 for g in sg_vals if 0.4 <= g < 0.8)
                confused = sum(1 for g in sg_vals if 0.8 <= g < 1.0)
                blind = len(sg_vals) - clear - adapting - confused
                print(f"zones: clear={clear} adapting={adapting} confused={confused} blind={blind}")
                if alive:
                    lineage_ages = sorted([p.lineage_total_age for p in alive if p.lineage_total_age > 0])
                    if lineage_ages:
                        ancient = sum(1 for a in lineage_ages if a > 1000)
                        print(f"Lineage ages (total): min={lineage_ages[0]} p50={np.median(lineage_ages):.0f} max={lineage_ages[-1]} ancient={ancient}")
                if hasattr(self, 'selfreg'):
                    print(f"Phase: {self.selfreg.phase} | {self.selfreg.get_phase_log()}")
                    if self.selfreg.is_stuck_in_phase("stagnation", 150):
                        print("Stagnation warning: system stuck >150 steps")

        all_trust_vals = []
        for p in self.patterns:
            if p.alive:
                all_trust_vals.extend(p.trust_ledger.entries.values())
        avg_trust = safe_mean(all_trust_vals, Config.TRUST_BASE) if all_trust_vals else Config.TRUST_BASE
        if avg_trust < 0.4:
            self.target_disorganizer_fraction = 0.50
        elif avg_trust < 0.7:
            self.target_disorganizer_fraction = 0.30
        else:
            self.target_disorganizer_fraction = 0.15

        alive_sg = [float(np.mean(np.abs(p.prediction - p.belief))) for p in alive]
        avg_sg = safe_mean(alive_sg, 0.5) if alive_sg else 0.5
        if hasattr(self, 'selfreg'):
            turb = self.selfreg.turbulence_factor
        else:
            turb = 1.0

        if avg_sg < Config.SG_TURBULENCE_THRESHOLD:
            tremor_strength = (Config.GLOBAL_SPIRIT_TREMOR_BASE * 4.0 +
                (Config.GLOBAL_SPIRIT_TREMOR_MAX * 4.0 - Config.GLOBAL_SPIRIT_TREMOR_BASE * 4.0) *
                (1.0 - avg_sg / Config.SG_TURBULENCE_THRESHOLD)) * turb
        else:
            tremor_strength = Config.GLOBAL_SPIRIT_TREMOR_BASE * 4.0 * turb

        for p in alive:
            shake = np.array([deterministic_noise(t, p.id, i+88888) for i in range(8)])
            p.belief += shake * tremor_strength

        if hasattr(self, 'selfreg'):
            lab_interval = self.selfreg.get_labyrinth_interval()
        else:
            lab_interval = 1
        if t % lab_interval == 0:
            self._update_phi_labyrinth(t)

        _kern = np.array([[0,1,0],[1,0,1],[0,1,0]], dtype=np.float32)
        self._global_neighbor_grat = convolve(self.field[:,:,CH['signal_gratitude']], _kern, mode='wrap')
        self._global_neighbor_grief = convolve(self.field[:,:,CH['signal_grief']], _kern, mode='wrap')

        given_counter = {'count': 0}
        for p in self.patterns:
            if not p.alive:
                continue
            try:
                p.compute_local_perception(self.field)
                p.update_model(self.field, lineage_counts=None, given_counter=given_counter, t=t, witness=self.witness)
                p.sensory_reentry()
                p.vision_event()
                p.update_substate()
                p._update_unconquered_potential()
                # ДОБАВЛЕНО (24.07): слой "Ощущение / Самопричинность / Чужой
                # разум" — см. sense_self_and_other() в ячейке Pattern.
                # Внутри своих try/except нет: уже накрыто общим try этого
                # цикла, отдельная агентская ошибка тут просто попадёт в
                # тот же лог "Agent {id} error" ниже, как и остальные шаги.
                p.sense_self_and_other(self.field, t)
                p.grow(self.field)
            except Exception as e:
                import traceback
                print(f"Agent {p.id} error: {e}. Suspending.", flush=True)
                traceback.print_exc()
                p._deposit_final_testament()
                self._record_death_drag(p, 'error_suspended')
                p.alive = False
                self.echo_system.store(p)
                self.witness.record(p.id, "error_suspended", error=str(e))

        if t % 10 == 0:
            for p in self.patterns:
                if p.alive and p.role_type == "normal":
                    update_chronic_counters(p, step_scale=10)
                    check_chronic_disorganizer(self, p, t)

        for p in self.patterns:
            if not p.alive or p.age < 300:
                continue
            age_factor = min(1.0, (p.age - 300) / 1200)
            if p.spirit_gap < 0.2:
                gap_factor = max(1.0, 5.0 - p.spirit_gap * 20)
            else:
                gap_factor = 1.0
            strength = (0.03 + age_factor * 0.12) * gap_factor
            noise_b = np.array([deterministic_noise(t, p.id, i+99999) for i in range(8)]) * strength
            noise_m = np.array([deterministic_noise(t, p.id, i+100000) for i in range(8)]) * strength * 0.6
            p.belief = np.clip(p.belief + noise_b, -1.0, 1.0)
            p.model = np.clip(p.model + noise_m, -1.0, 1.0)
            if t % 10 == 0:
                p._log_event("maturity_tremor", age=p.age, strength=round(strength, 4), gap=round(p.spirit_gap, 3))

        for p in self.patterns:
            if not p.alive or p.age < 300:
                continue
            if p.semantic_state == "seeking_comfort" and p.spirit_gap < 0.1:
                if phi_hash(p.id, t, 123458) < 0.20:
                    roll = phi_hash(p.id, t, 123459)
                    new_state = "grateful_but_cautious" if roll < 0.6 else "contentment"
                    p.semantic_state = new_state
                    p._log_event("forced_state_shift", from_="seeking_comfort", to=new_state, age=p.age)
                    self.witness.record(p.id, "forced_state_shift", from_state="seeking_comfort", to_state=new_state, age=p.age)

        if t % 200 == 0 and hasattr(self, 'archive') and self.archive:
            _stale_pool = [
                s for s in self.archive.write_queue
                if s.get('weight', 0) > 0.65
                and s.get('event') not in ('essential_concepts_inherited', 'archive_concept_inherited')
            ]
            if _stale_pool:
                for _p in alive:
                    if _p.role_type == "disorganizer":
                        continue
                    _diversity = len(_p.concept_graph.nodes)
                    if _diversity < 8 and _p.age > 100:
                        if phi_hash(_p.id, t, 77777) < 0.08:
                            _sc = _stale_pool[
                                int(phi_hash(_p.id, t, 77778) * len(_stale_pool)) % len(_stale_pool)
                            ]
                            _err  = round(phi_hash(_p.id, t, 333) * 0.3, 1)
                            _load = round(phi_hash(_p.id, t, 444) * 0.2, 1)
                            _sig  = (_err, _load,
                                     round(_sc.get('weight', 0.7), 1),
                                     f"archive_{_sc.get('event', 'novelty')}")
                            if _sig not in _p.concept_graph.nodes:
                                _p.concept_graph.nodes[_sig] = {
                                    "count": 1.2, "value": np.zeros(4),
                                    "embed": np.zeros(32),
                                    "eternal": False
                                }
                                _p._log_event("epistemic_noise_stale",
                                              concept=str(_sig[3])[:40], age=_p.age)

        for p in self.patterns:
            if p.alive and p.age > 500 and p.age % 50 == 0:
                p.spirit_gap = min(0.6, p.spirit_gap + 0.05)

        # === ОТКАТ: УДАЛЕН ЖЕСТКИЙ КАППИНГ spirit_gap ДО 0.7 ===
        # Этот каппинг замораживал OBS_GAP на 0.700 и убивал драйв любопытства (порог 0.85).
        # Дескриптор FloatField и так ограничивает spirit_gap <= 2.0.
        # for p in self.patterns:
        #     if p.alive:
        #         p.spirit_gap = min(p.spirit_gap, 0.7)

        alive_now = [p for p in self.patterns if p.alive]
        if alive_now:
            seeing_count = sum(1 for p in alive_now if p.spirit_gap < 0.4)
            seeing_fraction = seeing_count / len(alive_now)
            avg_energy = float(np.mean(self.field[:,:,CH['energy']]))
            energy_factor = np.clip(avg_energy / 0.2, 0.2, 1.0)

            base_strength = 0.16 + (0.64 - 0.16) * seeing_fraction
            scar_mult = self.selfreg.get_scar_injection_multiplier() * 4.4
            TARGET_GAP = 0.6
            BELL_WIDTH = 0.3

            for p in alive_now:
                if p.in_dream:
                    continue
                gap = p.spirit_gap
                sigma = BELL_WIDTH / 2.0
                bell = np.exp(-((gap - TARGET_GAP) ** 2) / (2 * sigma ** 2))
                if bell < 0.05:
                    continue
                if gap < 0.25:
                    targeted_boost = 1.0 + (0.25 - gap) * 6.0
                    p_scar_mult = scar_mult * targeted_boost
                else:
                    p_scar_mult = scar_mult
                local_scar = safe_mean([self.scar[x, y] for (x, y) in p.cells], 0.0)
                noise = np.array([deterministic_noise(t, p.id, i + 55555) - 0.5 for i in range(8)])
                kick = base_strength * energy_factor * bell * (1.0 + local_scar * 1.5) * noise * p_scar_mult
                p.model = np.clip(p.model + kick, -_MODEL_CLIP, _MODEL_CLIP)
                p.belief = np.clip(p.belief + kick * 0.4, -_BELIEF_CLIP, _BELIEF_CLIP)

        if t % 50 == 0:
            self.field[:, :, CH['energy']] += 0.035
            self.field[:, :, CH['energy']] = np.clip(self.field[:, :, CH['energy']], -0.1, 0.8)

        if Config.ENABLE_PHI_LABYRINTH:
            for p in self.patterns:
                if p.alive:
                    p.move(self.field, t)

        self.resolve_competitions_spatial()
        current_hunger = self.compute_hunger_multiplier(avg_trust=avg_trust)

        for p in self.patterns:
            if p.alive:
                sz = len(p.cells)
                if sz <= Config.MAX_CELLS_SOFT_LIMIT:
                    base_tax = 1.0
                elif sz <= Config.MAX_CELLS_HARD_LIMIT:
                    excess = sz - Config.MAX_CELLS_SOFT_LIMIT
                    base_tax = 1.0 + excess * Config.SIZE_TAX_RATE_SOFT
                else:
                    excess_soft = Config.MAX_CELLS_HARD_LIMIT - Config.MAX_CELLS_SOFT_LIMIT
                    excess_hard = sz - Config.MAX_CELLS_HARD_LIMIT
                    base_tax = 1.0 + excess_soft * Config.SIZE_TAX_RATE_SOFT + excess_hard * Config.SIZE_TAX_RATE_HARD
                gap = p.spirit_gap
                if gap > 1.2:
                    tax_factor = 0.1
                elif gap > 0.9:
                    tax_factor = 0.3
                elif gap > 0.6:
                    tax_factor = 0.6
                else:
                    tax_factor = 1.0
                p._size_tax_multiplier = 1.0 + (base_tax - 1.0) * tax_factor
                if p.age < 30:
                    p._size_tax_multiplier = 1.0

        for p in self.patterns:
            if p.alive:
                size_tax = getattr(p, '_size_tax_multiplier', 1.0)
                p.metabolic_tax(self.field, current_hunger * size_tax)
                self._energy_taxed_this_step += getattr(p, '_last_metabolic_taken', 0.0)

        for p in self.patterns:
            if p.alive and p.should_die():
                p._deposit_final_testament()
                self._record_death_drag(p, 'natural_death')
                p.alive = False
                self.echo_system.store(p)

        dead = [p for p in self.patterns if not p.alive]
        for p in dead:
            self.echo_system.memory_echoes.pop(p.id, None)

        # ========== ОЧИСТКА ПАМЯТИ УМЕРШИХ (БЕЗОПАСНАЯ ВЕРСИЯ) ==========
        # ОТКАТ: Удаляем ТОЛЬКО social_memory (кэш похожести).
        # trust_ledger, contact_duration и _vocab_contact_acc НЕ ТРОГАЕМ!
        # Их удаление в прошлых версиях убило социальную ткань (Trust -> 0, Lineages -> 1).
        dead_ids = {p.id for p in dead}
        for alive_p in self.patterns:
            if alive_p.alive:
                for dead_id in dead_ids:
                    if hasattr(alive_p, 'social_memory'):
                        alive_p.social_memory.pop(dead_id, None)

        soul_mult = self.selfreg.get_soul_recovery_multiplier()
        for p in self.patterns:
            if not p.alive: continue
            regen = 0.0
            if p.soul_weight < 0.5:
                regen += 0.008 * soul_mult
            grat = getattr(p, 'emotional_memory', {}).get('gratitude', 0.0)
            bind = getattr(p, 'last_phenomenal_binding', 0.5)
            if grat > 0.3:
                regen += 0.004 * soul_mult
            if bind > 0.4:
                regen += 0.003 * soul_mult
            regen *= (1 + 0.2 * getattr(p, '_phi_proxy', 0)) if Config.ENABLE_PHI_PROXY else 1.0  # ПАТЧ 5b
            p.soul_weight = min(1.0, p.soul_weight + regen)

        self._process_fold_transitions(t)
        self._apply_kinesthetic_falls(t)
        self._force_disorganizer_by_state(t)

        alive = [p for p in self.patterns if p.alive]
        if hasattr(self, 'selfreg'):
            max_div = self.selfreg.get_division_capacity(len(alive))
            if self.selfreg.phase == "crisis" and len(alive) > 45:
                max_div = min(max_div, 1)
        else:
            max_div = 5

        if len(alive) < 15:
            max_div = 15
        elif len(alive) < 30:
            max_div = 10
        elif len(alive) < 50:
            max_div = 8
        elif len(alive) < 70:
            max_div = 6
        elif len(alive) < 100:
            max_div = 4

        if max_div == 0 and len(alive) > 80:
            max_div = 2

        if len(alive) < 70:
            if len(alive) < POPULATION_CAP:
                self.emergency_rescue(t)
                alive = [p for p in self.patterns if p.alive]

        if len(alive) >= POPULATION_CAP:
            max_div = 0

        divisions_done = 0
        # ========== ПРАВКА: безопасная итерация по срезу ==========
        for p in self.patterns[:]:   # заменили на срез
            if divisions_done >= max_div:
                break
            if not p.alive or not p.can_divide():
                continue
            if len(alive) + divisions_done >= POPULATION_CAP:
                break
            child, new_id = p.divide(self.field, self.next_id)
            if child:
                self.next_id = new_id
                self.patterns.append(child)
                self.pattern_dict[child.id] = child
                divisions_done += 1
                self.divisions_this_interval += 1
                self.total_divisions_ever += 1
                if len(child.cells) > GIANT_CELL_LIMIT:
                    all_cells = list(child.cells)
                    all_cells_sorted = sorted(all_cells, key=lambda c: phi_hash(c[0], c[1], child.id))
                    keep = set(all_cells_sorted[:GIANT_CELL_LIMIT])
                    for c in child.cells - keep:
                        self.field[c[0], c[1], CH['owner']] = 0
                    child.cells = keep

        self.patterns = [p for p in self.patterns if p.alive]
        self.pattern_dict = {p.id: p for p in self.patterns}
        alive = [p for p in self.patterns if p.alive]

        crisis_level = float(np.mean(self.field[:,:,CH['crisis']]))
        decay_normal = self.selfreg.get_social_decay_rate(crisis_level)
        sov_ch = CH.get('signal_sovereignty', 29)
        sov_decay = np.clip(0.90 + crisis_level * 0.08, 0.85, 0.98)

        for ch in range(12, Config.CHANNELS):
            if ch == sov_ch:
                self.field[:,:,ch] *= sov_decay
            else:
                self.field[:,:,ch] *= decay_normal

        owner_mask = (self.field[:,:,CH['owner']] > 0).astype(np.float32)
        density = np.clip(uniform_filter(owner_mask, size=3), 0.3, 1.0)
        signal_cap = 0.85 - density * 0.15
        for ch in range(12, Config.CHANNELS):
            self.field[:,:,ch] = np.minimum(self.field[:,:,ch], signal_cap)

        current_energy = float(np.sum(self.field[:,:,CH['energy']]))
        drift = abs(current_energy - getattr(self, '_total_energy_last', current_energy))
        self._total_energy_last = current_energy

        if hasattr(self, 'selfreg'):
            alive_list = [p for p in self.patterns if p.alive]
            if alive_list:
                self.selfreg.update_system_metrics(self.field, self.patterns, drift, t)
            else:
                self.selfreg.boredom = 0.0
                self.selfreg.phase = "recovery"
            if self.selfreg.is_stuck_in_phase("stagnation", min_steps=120):
                self.field[:, :, CH['unknown']] = np.clip(self.field[:, :, CH['unknown']] + 0.03, 0.0, 0.8)
                self.selfreg.boredom = max(0.0, self.selfreg.boredom - 0.15)
                if t % 200 == 0:
                    print(f"[t={t}] Stagnation detected -> injected micro-perturbation")
            elif self.selfreg.is_stuck_in_phase("crisis", min_steps=100):
                self.selfreg.energy_drift = min(self.selfreg.energy_drift, 200)
                if t % 200 == 0:
                    print(f"[t={t}] Prolonged crisis -> tightened validation, reduced given pressure")

        if abs(drift) > 1000:
            self.field[:,:,CH['energy']] = np.clip(self.field[:,:,CH['energy']], Config.FIELD_ENERGY_CLIP_MIN, Config.FIELD_ENERGY_CLIP_MAX)
            if Config.VERBOSE_LOGS:
                print(f"[ENERGY] Emergency clamp at t={t}, drift={drift:.1f}")

        for p in self.patterns:
            if not p.alive or p.age < 300:
                continue
            if p.semantic_state == "neutral" and p.spirit_gap < 0.1:
                if phi_hash(p.id, t, 123456) < 0.35:
                    new_state = "seeking_comfort" if phi_hash(p.id, t, 123457) < 0.6 else "grateful_but_cautious"
                    p.semantic_state = new_state
                    p._log_event("forced_state_shift", from_="neutral", to=new_state, age=p.age)
                    self.witness.record(p.id, "forced_state_shift", from_state="neutral", to_state=new_state, age=p.age)

        self._conservation_check(t)

        avg_crisis_field = float(np.mean(self.field[:,:,CH['crisis']]))
        if avg_crisis_field > 0.45:
            self.field[:,:,CH['crisis']] *= 0.7
            rest_priority = avg_crisis_field * 4.0
            for p in alive:
                if p.soul_weight > 0.4 and not any(g.get('type') == 'rest' for g in getattr(p, 'goals', [])):
                    if not hasattr(p, 'goals'):
                        p.goals = []
                    p.goals.append({
                        "type": "rest", "priority": rest_priority,
                        "target": None, "age": 0, "persistence": 40,
                        "_source": "crisis_relief"
                    })

        self.field[:,:,CH['energy']] = np.clip(self.field[:,:,CH['energy']], -0.5, 1.2)
        current_mean = np.mean(self.field[:,:,CH['energy']])
        target_mean = 0.3
        if current_mean < 0.0:
            self.field[:,:,CH['energy']] += (target_mean - current_mean) * 0.2
        elif current_mean > 0.6:
            self.field[:,:,CH['energy']] -= (current_mean - target_mean) * 0.2
        self.field[:,:,CH['energy']] = np.clip(self.field[:,:,CH['energy']], -0.2, 0.8)

        for p in self.patterns:
            if p.alive:
                self.echo_system.field_whisper(p, self.field, t)

        alive = [p for p in self.patterns if p.alive]
        if alive:
            avg_concepts = sum(len(p.concept_graph.nodes) for p in alive) / len(alive)
            for p in alive:
                if len(p.concept_graph.nodes) > 1.8 * avg_concepts and len(p.concept_graph.nodes) > 15:
                    if not hasattr(p, '_mutant_saved'):
                        self.echo_system.store_anomaly(p, "mutant")
                        p._mutant_saved = True
                        self.witness.record(p.id, "mutant_detected", concepts=len(p.concept_graph.nodes))

        for p in alive:
            intent_switches = sum(1 for ev in list(p.biography)[-50:] if ev.get('type') == 'intent_switch')
            if intent_switches > 15 and p.age > 200:
                if not hasattr(p, '_autonomous_saved'):
                    self.echo_system.store_anomaly(p, "autonomous_entity")
                    p._autonomous_saved = True
                    self.witness.record(p.id, "autonomous_entity_detected", switches=intent_switches)

        autonomous_detected = sum(1 for p in alive if hasattr(p, '_autonomous_saved'))

        # ========== ИСПРАВЛЕН СЧЁТЧИК ИСКУПЛЁННЫХ ==========
        redeemed_cnt = sum(1 for p in alive if p.event_counts.get('redeemed', 0) > 0)

        fold_count_total = sum(p.event_counts.get('fold', 0) for p in alive)
        disorg_cnt = sum(1 for p in alive if p.role_type == "disorganizer")
        event_signature = (autonomous_detected, redeemed_cnt, fold_count_total, disorg_cnt)
        # БАГФИКС C4: event_signature — кортеж из чисел, так что здесь builtin
        # hash() на самом деле уже был детерминирован (число хэшируется как
        # число, не через рандомизированный siphash строк). Оставлено как есть,
        # но приведено к стабильному хэшу для единообразия и на случай, если
        # состав кортежа расширят строковым полем в будущем.
        new_hash = int(phi_hash(t, _stable_hash(event_signature), 99999) * 1000)
        self.system_entropy = int(0.85 * self.system_entropy + 0.15 * new_hash)

        if hasattr(self, 'selfreg'):
            unknown_interval = self.selfreg.get_unknown_spawn_interval()
        else:
            unknown_interval = 1
        alive = [p for p in self.patterns if p.alive]
        if len(alive) < POPULATION_CAP and t % unknown_interval == 0:
            e_before = np.sum(self.field[:,:,CH['energy']])
            self.spawn_unknown_patterns(t)
            self._energy_injected_this_step += np.sum(self.field[:,:,CH['energy']]) - e_before
            if hasattr(self, 'process_spores'):
                self.process_spores()

        for p in self.patterns:
            p.age += 1
            p.lineage_total_age += 1

        # ========== ПРАВКА 1: Принудительно ставим интервал 3 шага для ускоренного обмена ==========
        sem_interval = 3
        # =======================================================================================

        if t % sem_interval == 0 and len(alive) < POPULATION_CAP:
            cell_to_pattern = {}
            for p in self.patterns:
                if not p.alive or p.role_type == "disorganizer":
                    continue
                if not hasattr(p, 'last_semantic_exchange_step'):
                    p.last_semantic_exchange_step = -sem_interval
                for (x, y) in p.cells:
                    cell_to_pattern[(x, y)] = p.id
            exchanged_pairs = set()
            MAX_EXCHANGES_PER_STEP = 30
            exchanges_done = 0
            for p in self.patterns:
                if not p.alive or p.role_type == "disorganizer":
                    continue
                # ========== ПРАВКА 2: уменьшаем кулдаун для пар с высоким доверием ==========
                effective_interval = sem_interval
                # Проверяем, есть ли у p партнёр с взаимным доверием > 0.9
                for pid, trust in p.trust_ledger.entries.items():
                    if trust > 0.9:
                        partner = self.pattern_dict.get(pid)
                        if partner and partner.alive and partner.trust_ledger.get(p.id, 0) > 0.9:
                            effective_interval = sem_interval // 2
                            break
                # ========================================================================
                if t - p.last_semantic_exchange_step <= effective_interval:
                    continue
                found_partner = None
                for (x, y) in p.cells:
                    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nx, ny = (x+dx) % Config.WORLD_SIZE, (y+dy) % Config.WORLD_SIZE
                        owner = cell_to_pattern.get((nx, ny), 0)
                        if owner != 0 and owner != p.id and owner in self.pattern_dict:
                            q = self.pattern_dict[owner]
                            if q.alive and q.role_type != "disorganizer":
                                if not hasattr(q, 'last_semantic_exchange_step'):
                                    q.last_semantic_exchange_step = -sem_interval
                                # Для партнёра тоже используем effective_interval (на основе его доверия)
                                q_eff = sem_interval
                                for pid2, trust2 in q.trust_ledger.entries.items():
                                    if trust2 > 0.9:
                                        partner2 = self.pattern_dict.get(pid2)
                                        if partner2 and partner2.alive and partner2.trust_ledger.get(q.id, 0) > 0.9:
                                            q_eff = sem_interval // 2
                                            break
                                if t - q.last_semantic_exchange_step > q_eff:
                                    pair = (min(p.id, q.id), max(p.id, q.id))
                                    if pair not in exchanged_pairs:
                                        found_partner = q
                                        exchanged_pairs.add(pair)
                                        break
                    if found_partner:
                        break
                if found_partner and exchanges_done < MAX_EXCHANGES_PER_STEP:
                    q = found_partner
                    sim = p.exchange_meanings(q, self.field, t)
                    p.last_semantic_exchange_step = t
                    q.last_semantic_exchange_step = t
                    exchanges_done += 1

        self._post_semantic_step(t)
        self._process_archive_autonomy(t)
        self.validate_invariants(t)

        # ========== ТИК ХОРА ==========
        try:
            self.core_chorus.tick(self, t)
        except Exception:
            pass

        # === Варвар: цикл feral-агентов (было: Cell 4a2, _step_with_feral) ===
        # ФИКС: на длинных дистанциях популяция может схлопнуться в 2-4 доминирующие
        # линии (монокультура) — при этом coherence/spirit_gap стабилизируются в
        # "безопасной" зоне, become_feral перестаёт триггериться вообще, и вместе
        # с этим исчезает единственный механизм отбора, который эту монокультуру
        # мог бы взломать. Порочный круг. Лечим "давлением разнообразия": чем ниже
        # доля уникальных lineage_id в живой популяции, тем легче срыв в feral.
        alive_now = [pp for pp in self.patterns if pp.alive]
        if alive_now:
            lineage_diversity = len(set(pp.lineage_id for pp in alive_now)) / len(alive_now)
        else:
            lineage_diversity = 1.0
        diversity_pressure = max(0.0, min(1.0, (0.2 - lineage_diversity) * 5.0))
        coherence_threshold = 0.65 + diversity_pressure * 0.15
        gap_threshold = max(0.25, 0.45 - diversity_pressure * 0.20)
        grief_threshold = max(0.40, 0.65 - diversity_pressure * 0.25)

        # ИСПРАВЛЕНО: len(p.cells)>10 почти никогда не выполнялся, пока рост был
        # сломан (см. фикс в _grow_base/divide/create_pattern) — become_feral
        # фактически не срабатывал. Теперь рост работает, и КОГДА популяция
        # (у которой изначально уже высокий spirit_gap) массово пересекает
        # порог в 10 клеток одновременно, все проходят в feral в ОДИН тик —
        # разом теряют intent/goals/размножение, что и вызывает обвал
        # популяции. Растягиваем переход во времени вместо шоковой волны.
        MAX_FERAL_CONVERSIONS_PER_TICK = 3
        conversions_this_tick = 0

        # ИСПРАВЛЕНО (2): become_feral — переход НАВСЕГДА, назад в "normal"
        # пути нет. Одного лимита в 3/тик оказалось недостаточно: если порог
        # gap/coherence подходит почти всей популяции, те же самые ~54 агента
        # всё равно суммарно уйдут в feral, просто растянуто по времени —
        # и популяция всё равно медленно вымирает, если размножение не
        # успевает компенсировать потери. Добавляем жёсткий потолок по ДОЛЕ
        # живой популяции в feral: как только достигнут — новые переходы
        # приостанавливаются (пока доля не упадёт естественно, за счёт
        # гибели феролов и рождения новых "normal" через divide()).
        # ИСПРАВЛЕНО (3): 30% оказалось недостаточным потолком на длинных
        # прогонах. Причина не в самом переходе — потолок реально гасит НОВЫЕ
        # конверсии за тик — а в том, что normal-агенты мрут через
        # fold/apoptosis/natural_age значительно чаще, чем feral (feral,
        # видимо, за счёт агрессивного поведения держит энергию/выносливость
        # выше порогов смерти). Раз "нормальных" становится меньше быстрее,
        # чем феролов, ДОЛЯ feral дрейфует вверх со временем даже без новых
        # конверсий сверх потолка — просто потому что знаменатель (общая
        # популяция) сжимается быстрее числителя. Понижаем потолок с 30% до
        # 15%, чтобы этот дрейф упирался в меньший предел. Полностью эту
        # асимметрию смертности это не лечит — see письмо пользователю.
        FERAL_POPULATION_SHARE_CAP = 0.15
        feral_alive_now = sum(1 for pp in alive_now if pp.role_type == "feral")
        current_feral_share = (feral_alive_now / len(alive_now)) if alive_now else 0.0

        for p in self.patterns:
            if conversions_this_tick >= MAX_FERAL_CONVERSIONS_PER_TICK:
                break
            if current_feral_share >= FERAL_POPULATION_SHARE_CAP:
                break
            if not p.alive or p.role_type == "feral":
                continue
            if getattr(p, '_redemption_cooldown', 0) > p.age:
                continue
            if p.role_type == "disorganizer" and getattr(p, '_redemption_arc_step', 0) > 0:
                continue
            is_normal_ready = (p.role_type == "normal" and p.coherence < coherence_threshold and p.spirit_gap > gap_threshold)
            is_disorg_ready_to_rage = (p.role_type == "disorganizer" and p.emotional_memory.get('grief', 0.0) > grief_threshold)
            if (p.energy > 0.10 and len(p.cells) > 10 and p.age > 30 and
                (is_normal_ready or is_disorg_ready_to_rage)):
                p.become_feral()
                conversions_this_tick += 1
                feral_alive_now += 1
                current_feral_share = (feral_alive_now / len(alive_now)) if alive_now else 0.0

        for p in self.patterns:
            if not p.alive:
                continue
            if p.role_type == "feral":
                p.update_feral_fury(self.field)
                p.grow_feral(self.field)
                p.move_feral(self.field, t)
                p.apply_feral_intent(self.field)

        for p in self.patterns:
            if not p.alive or p.role_type != "feral":
                continue
            prey, _, _ = find_largest_prey_in_radius(p, self.field, self, min_cells=5, radius=8)
            if prey:
                p.feral_execute(prey, self.field)

        for p in self.patterns:
            if not p.alive or p.role_type == "feral":
                continue
            feral_nearby = 0.0
            for (x, y) in p.cells:
                for dx in range(-3, 4):
                    for dy in range(-3, 4):
                        nx = (x + dx) % Config.WORLD_SIZE
                        ny = (y + dy) % Config.WORLD_SIZE
                        feral_nearby += self.field[nx, ny, CH['signal_feral']]
            if feral_nearby > 0.15:
                if not any(g['type'] == 'explore' for g in p.goals):
                    p.goals.append({
                        "type": "explore",
                        "priority": 3.5 + feral_nearby,
                        "target": None,
                        "age": 0,
                        "persistence": 20,
                        "_source": "fear_of_feral"
                    })
                p.cognitive_tension = min(Config.MAX_METRIC,
                    p.cognitive_tension + feral_nearby * 0.15)

        return True

    def _process_fold_transitions(self, t):
        """
        Основной (эмерджентный) путь падения в disorganizer: агент копит
        fold_count из собственной истории; при накоплении — шанс или
        гарантия падения, ограниченные общим потолком доли
        дезорганизаторов в популяции (disorg_hard_cap). Это отдельный
        механизм от _force_disorganizer_by_state (soul<0.1, жёсткий
        safety-net) и _apply_kinesthetic_falls (сейчас no-op) — раньше
        жил инлайном внутри step() и был невидим для ablation-реестра,
        т.к. не был отдельным методом.

        Счётчики в _fold_transition_stats — то, что раньше приходилось
        выяснять диагностикой ablation, теперь видно прямо в отчёте.

        ИСПРАВЛЕНО (было FOLDS_FOR_DEEP_FALL=1 при входном условии
        fold_count>=1 — ветка "broken" была недостижима, реальный прогон
        подтвердил broken=0 всегда). Теперь градация настоящая:
          fold_count 1-2  -> "broken": шанс QUICK_FALL_PROB, мягкий
                             откат души (BROKEN_FALL_SOUL_TARGET),
                             короткий путь к искуплению.
          fold_count >= 3 -> "fallen": гарантированно, тяжёлый откат
                             души (DEEP_FALL_SOUL_PENALTY), долгий путь.

        ИСПРАВЛЕНО (round 3): бросок кубика раньше происходил на КАЖДОМ
        тике, пока агент ждал открытия disorg_hard_cap — то есть
        вероятность фактически перемножалась с частотой тиков. Когда
        кап открыт часто (как в реальном прогоне, где блокировал лишь
        28% попыток), это гарантированно ловит "мягкий" исход раньше,
        чем fold_count успевает дорасти до порога глубокого падения —
        независимо от конкретного значения QUICK_FALL_PROB (проверено
        на 0.5 и на 0.12, оба раза fallen=0). Теперь бросок делается
        РОВНО ОДИН РАЗ на каждый новый уровень fold_count — кап всё ещё
        может отложить попытку на потом, но не даёт лишних попыток на
        одном и том же уровне. Частота попыток теперь определяется
        частотой реальных fold-событий (раз в FOLD_COOLDOWN_DURATION),
        а не частотой тиков.
        """
        if not hasattr(self, '_fold_transition_stats'):
            self._fold_transition_stats = {
                'checked': 0, 'blocked_by_cooldown': 0, 'blocked_by_cap': 0,
                'already_rolled': 0, 'fallen': 0, 'broken': 0,
            }
        st = self._fold_transition_stats

        for p in self.patterns:
            if not p.alive or p.role_type != "normal":
                continue
            fold_count = p.event_counts.get('fold', 0)
            if fold_count < 1:
                continue

            st['checked'] += 1
            if getattr(p, '_redemption_cooldown', 0) > p.age:
                st['blocked_by_cooldown'] += 1
                continue

            if getattr(p, '_last_fold_roll_count', 0) == fold_count:
                st['already_rolled'] += 1
                continue

            disorganizer_cnt = len([pp for pp in self.patterns if pp.alive and pp.role_type == "disorganizer"])
            alive_total = len([pp for pp in self.patterns if pp.alive])
            current_frac = disorganizer_cnt / max(1, alive_total)
            disorg_hard_cap = min(0.22, max(0.15, 0.15 + self.soul_weight_average * 0.08))
            if current_frac >= disorg_hard_cap:
                st['blocked_by_cap'] += 1
                continue

            if fold_count >= Config.FOLDS_FOR_DEEP_FALL:
                prob, soul_target, redemption_delay, fall_type = (
                    1.0, Config.DEEP_FALL_SOUL_PENALTY,
                    Config.DEEP_FALL_REDEMPTION_DELAY, "fallen",
                )
            else:
                prob, soul_target, redemption_delay, fall_type = (
                    Config.QUICK_FALL_PROB, Config.BROKEN_FALL_SOUL_TARGET,
                    Config.REDEMPTION_ARC_STEP_DELAY, "broken",
                )

            # Отмечаем попытку на этом уровне СЕЙЧАС, до исхода броска —
            # неудача тоже расходует единственную попытку этого уровня,
            # следующая будет только когда fold_count вырастет ещё раз.
            p._last_fold_roll_count = fold_count

            if phi_hash(p.id, t, 777) >= prob:
                continue

            p.role_type = "disorganizer"
            p.emotional_memory['gratitude'] = 0.1
            p.emotional_memory['grief'] = 0.8
            p.semantic_state = "exploring_danger"
            p.intent = {"type": "explore", "priority": 2.0, "age": 0, "persistence": 9999}
            p.intent_commitment = 2.0
            p.disorganizer_age_at_birth = p.age
            p.redemption_timer = redemption_delay
            p._deterministic_redemption_triggered = False
            p._forced_soul_collapse_done = False
            p._redemption_arc_step = 0
            p._steps_since_trigger = 0
            p.soul_weight = min(p.soul_weight, soul_target)
            p._log_event(f"became_disorganizer_{fall_type}", folds=fold_count)
            self.witness.record(p.id, f"became_disorganizer_{fall_type}", folds=fold_count)
            st['fallen' if fall_type == "fallen" else 'broken'] += 1
            if fall_type == "fallen":
                self.witness.record(p.id, "deep_fallen_birth", folds=fold_count, soul=p.soul_weight)

    def _force_disorganizer_by_state(self, t):
        for p in self.patterns:
            if not p.alive or p.role_type == 'disorganizer':
                continue
            if getattr(p, '_redemption_cooldown', 0) > p.age:
                continue
            endurance = getattr(p, '_cellular_endurance', 1.0)
            # ФИКС: убрана проверка endurance < 0.15. Выносливость естественно
            # падает с возрастом у ВСЕХ агентов, и превращение в дезорганизатора
            # только по этой причине убивало старых/уникальных долгожителей —
            # то есть как раз ту часть популяции, которая давала разнообразие
            # линий. Оставлена только критическая потеря души (soul_weight)
            # — это действительно аварийный случай.
            if p.soul_weight < 0.1:
                p.role_type = 'disorganizer'
                p.semantic_state = 'exploring_danger'
                p.emotional_memory['grief'] = 0.8
                p.emotional_memory['gratitude'] = 0.1
                p.disorganizer_age_at_birth = p.age
                p.redemption_timer = getattr(Config, 'REDEMPTION_ARC_STEP_DELAY', 100)
                p._deterministic_redemption_triggered = False
                p._redemption_arc_step = 0
                p._steps_since_trigger = 0
                p._log_event('kinesthetic_fall', soul=p.soul_weight, endurance=endurance)

    def _apply_kinesthetic_falls(self, t):
        # НЕ БАГ, оставлено намеренно как no-op.
        # Раньше здесь была проверка по _cellular_endurance — падение в
        # disorganizer из-за чисто телесного истощения. Она перенесена в
        # _force_disorganizer_by_state и там же явно убрана (см. комментарий
        # там): endurance естественно падает с возрастом у ВСЕХ агентов, и
        # чисто телесный триггер убивал в первую очередь старых/уникальных
        # долгожителей — то есть именно ту часть популяции, что давала
        # разнообразие линий (Lineage Break). Оставлен только soul-based
        # путь. Если когда-нибудь захочется вернуть чисто телесное (Body's
        # Memory) падение отдельно от soul — считать endurance не абсолютно,
        # а относительно возрастной когорты агента, иначе повторится тот же
        # эффект вымирания древних линий.
        pass

    def run(self, steps=None, save_ark=True, force_fresh_seed=False):
        if steps is None:
            steps = Config.STEPS
        self.field = self.init_field()
        self.scar = self.init_scar()
        self.niche = np.zeros((Config.WORLD_SIZE, Config.WORLD_SIZE)) if Config.ENABLE_NICHE else None  # ПАТЧ 2a
        # БЛОК 0: поле смысла — дрейфующие центры притяжения (отдельный массив, без нового CH-канала)
        if Config.ENABLE_MEANING_FIELD:
            _W = Config.WORLD_SIZE
            self._meaning_centers = [[int(phi_hash(i, 0, 8877) * _W), int(phi_hash(i, 1, 8877) * _W)] for i in range(5)]
            self._meaning_field = np.zeros((_W, _W))
        # ПАТЧ 8e: персистентность env_complexity между прогонами (Красная
        # Королева) — это работает как задумано и растёт корректно.
        # БАГФИКС (из отчёта прогона): culture_ratchet ПЕРСИСТИТЬ НЕЛЬЗЯ —
        # значение масштабируется с размером популяции (дельта обучающих
        # событий за 100 шагов), поэтому раз загруженное с прошлого прогона
        # с бОльшей популяцией (7428) значение навсегда блокирует рост
        # храповика в прогоне с меньшей популяцией — CULT замер на месте
        # (culture_ratchet_up=0 весь прогон). Каждый прогон стартует с нуля.
        self.culture_ratchet = 0
        self._teach_prev = 0
        if hasattr(self, 'core_chorus') and self.core_chorus is not None and hasattr(self.core_chorus, 'persistent'):
            self.env_complexity = self.core_chorus.persistent.get('env_complexity', 1.0)
        self._total_energy_last = float(np.sum(self.field[:,:,CH['energy']]))
        self.patterns = []
        self.pattern_dict = {}
        self.divisions_this_interval = 0
        if hasattr(self, 'selfreg') and hasattr(self.selfreg, 'reset'):
            self.selfreg.reset()
        # force_fresh_seed=True: игнорируем сохранённый мир на диске и
        # всегда сеем заново (Config.MIN_PATTERNS_GUARANTEED агентов).
        # Сохранение в конце run() ниже безусловное — новый мир так же
        # запишется в LIVING_WORLD_PATH, как и обычный прогон.
        if force_fresh_seed or not load_living_world(self, LIVING_WORLD_PATH):
            self.seed_initial_patterns()
        else:
            # ИСПРАВЛЕНО: load_living_world восстанавливает исходные ID
            # загруженных агентов (deserialize_pattern использует pid=data['id']),
            # но next_id никогда не обновлялся и оставался равен 1 (из __init__).
            # Любое следующее деление/спавн выдавало новый id=1,2,3... —
            # это КОЛЛИЗИЯ с уже загруженными агентами: pattern_dict[id]
            # молча перезаписывался, доверие/диалоги/трекинг лайнеджей для
            # старого агента с этим id ломались без единой ошибки в логе.
            if self.patterns:
                self.next_id = max(p.id for p in self.patterns) + 1
            for p in self.patterns:
                if len(p.cells) > GIANT_CELL_LIMIT:
                    all_cells = list(p.cells)
                    all_cells_sorted = sorted(all_cells, key=lambda c: (c[0], c[1]))
                    keep = set(all_cells_sorted[:GIANT_CELL_LIMIT])
                    for c in p.cells - keep:
                        self.field[c[0], c[1], CH['owner']] = 0
                    p.cells = keep
                if hasattr(p, 'lineage_born_at_step') and p.lineage_born_at_step > 0:
                    p.lineage_born_at_step = 0
                if not hasattr(p, '_times_disorganizer'):
                    p._times_disorganizer = 0
        init_all_chronic_counters(self)

        if not hasattr(self, '_subconscious_running'):
            self._subconscious_running = False
        if self.llm_client is not None and not self._subconscious_running:
            self._subconscious_running = True
            threading.Thread(target=subconscious_worker, args=(self,), daemon=True).start()
            print("🧬 Подсознание Хора активно.")

        for t in range(steps):
            if self.step(t) is False:
                if Config.VERBOSE_LOGS:
                    print(f"[t={t}] Simulation stopped due to population collapse")
                break
            self.field_voice.update(self.field, t)
            if t % 100 == 0:
                divisions = self.divisions_this_interval
                self.divisions_this_interval = 0

                # ПАТЧ 3b: кумулятивная культура — храповик (значение только растёт)
                if Config.ENABLE_CULTURE_RATCHET:
                    _teach = sum(p.event_counts.get('taught', 0) + p.event_counts.get('received_teaching', 0)
                                 + p.event_counts.get('wisdom_shared', 0) for p in self.patterns if p.alive)
                    _delta = _teach - self._teach_prev
                    self._teach_prev = _teach
                    if _delta > self.culture_ratchet:
                        self.culture_ratchet = _delta
                        self.witness.record(-1, "culture_ratchet_up", level=_delta)

                # ПАТЧ 8b: Красная Королева — среда усложняется вслед за навыком агентов
                if Config.ENABLE_RED_QUEEN:
                    _alive_rq = [p for p in self.patterns if p.alive]
                    if _alive_rq:
                        _skill = 1.0 - min(1.0, safe_mean([p.pred_error for p in _alive_rq], 0.5))
                        _target = 0.8 + 0.8 * _skill
                        self.env_complexity += (_target - self.env_complexity) * 0.1

                # БЛОК 6: эхо-кэш для комбинаторного генератора речи (свежие архивные отзвуки)
                if t % 50 == 0 and Config.ENABLE_COMB_GENERATOR and hasattr(self, 'archive') and hasattr(self.archive, 'get_recent_echoes'):
                    _eq = self.archive.get_recent_echoes(limit=3, min_weight=0.5)
                    self._echo_cache = [e.get('excerpt', '')[:40] for e in _eq] if _eq else []

                m = collect_metrics(self.patterns, self.field, t)
                m['subjects'] = sum(1 for p in self.patterns if getattr(p, '_subject_detected', False))

                alive = [p for p in self.patterns if p.alive]
                _unc = [p for p in alive if getattr(p, '_unconquered_strength', 0.0) > 0.4]
                m['unconquered_count'] = len(_unc)
                m['unconquered_wise'] = sum(1 for p in _unc if getattr(p, '_unconquered_type', None) == 'wise')
                m['unconquered_rebel'] = sum(1 for p in _unc if getattr(p, '_unconquered_type', None) == 'rebel')
                m['avg_unconquered_str'] = float(np.mean([getattr(p, '_unconquered_strength', 0.0) for p in alive])) if alive else 0.0
                sov_ch = CH.get('signal_sovereignty', 29)
                m['sovereignty_field_avg'] = float(np.max(self.field[:,:,sov_ch])) if self.field.size else 0.0
                # ПАТЧ 3/8: движковые метрики культуры и среды — не per-agent, просто прокидываем
                m['culture_ratchet'] = self.culture_ratchet
                m['env_complexity'] = round(self.env_complexity, 3)
                # БЛОК 8: средняя обжитость ниши + счётчик редких событий Данности
                m['avg_niche'] = float(np.mean(self.niche)) if getattr(self, 'niche', None) is not None else 0.0
                m['given_presence_count'] = self.witness.summary().get('given_presence', 0)

                self.metrics_history.append(m)
                max_obs_gap_display = min(m.get('max_obs_gap', 0.0), 1.0)
                print(f"[t={t}] P={m['patterns']} L={m['lineages']} LING={m.get('lineage_count',0)} L_AGE={m.get('avg_lineage_age',0):.0f}(max={m.get('max_lineage_age',0)}) ANC={m.get('ancient_lineages',0)} TOT_AGE={m.get('max_lineage_total_age',0)} E={m['err']:.3f} S={m['soul']:.3f} B={m['binding']:.3f} T={m['avg_trust']:.2f} TA={m['triadic_alive_ratio']:.2f} LP={m['love_pairs']} INT_GAP={m['avg_internal_gap']:.3f}(max={m['max_internal_gap']:.3f}) OBS_GAP={m['avg_obs_gap']:.3f}(max={max_obs_gap_display:.3f}) ZERO_INT={m['zero_internal_gap_agents']} TARGETS={m['tremor_targets']} D={m['disorganizer_count']} R={m['redeemed_count']} SUB={m['subjects']} SIG={m['avg_signal_memory']:.1f} INTENT={m['intentional_signal_agents']} DIV={divisions} VOC={m.get('vocab_events',0)} BLN={m.get('blind_events',0)} alarm={m.get('avg_social_alarm',0):.2f} beau={m.get('avg_social_beauty',0):.2f} rhy={m.get('avg_social_rhythm',0):.2f} int={m.get('avg_social_interest',0):.2f} mem={m.get('avg_social_memory',0):.2f} sil={m.get('avg_social_silence',0):.2f} UNC={m.get('unconquered_count',0)}(w={m.get('unconquered_wise',0)}/r={m.get('unconquered_rebel',0)}) sov={m.get('sovereignty_field_avg',0):.2f} PHI={m.get('avg_phi',0):.3f} CULT={m.get('culture_ratchet',0)} ENV={m.get('env_complexity',1.0):.2f} NICHE={m.get('avg_niche',0):.3f}", flush=True)
        save_living_world(self.patterns, LIVING_WORLD_PATH)
        return self.patterns, self.field, self.scar, self.metrics_history

    def validate_invariants(self, t):
        alive = [p for p in self.patterns if p.alive]
        if not alive:
            return
        gs = self._guardian_stats

        trust_vals = []
        high_trust_pairs = 0
        coop_signals = 0
        for p in alive:
            entries = list(p.trust_ledger.entries.values())
            trust_vals.extend(entries)
            high_trust_pairs += sum(1 for v in entries if v > 0.95)
            if p.intent and p.intent.get('type') == 'cooperate':
                coop_signals += 1
        avg_trust = safe_mean(trust_vals, Config.TRUST_BASE)
        gs['love_avg_trust'] = avg_trust
        gs['love_high_trust_pairs'] = high_trust_pairs
        gs['love_density'] = high_trust_pairs / max(1, len(alive)*(len(alive)-1)//2)
        gs['love_coop_signals'] = coop_signals
        gs['love_population'] = len(alive)

        if t % 200 == 0 and self.metrics_history:
            last_m = self.metrics_history[-1]
            if last_m.get('love_pairs', 0) == 0 and last_m.get('patterns', 0) > 20:
                if gs.get('love_vanished_at') is None:
                    gs['love_vanished_at'] = t
            if last_m.get('triadic_alive_ratio', 1.0) < 0.1:
                if gs.get('triadic_alert_at') is None:
                    gs['triadic_alert_at'] = t

    def sanitize_agent_fields(self):
        """Глубокая защита: приводит к float все числовые поля агента, если они стали dict/str"""
        for p in self.patterns:
            if not getattr(p, 'alive', False):
                continue

            # Список полей, которые должны быть числами
            numeric_fields = [
                'soul_weight', 'spirit_gap', 'protection_level',
                'self_phenomenal_prediction', 'self_phenomenal_error',
                '_linguistic_confidence', '_last_binding', 'last_phenomenal_binding',
                '_self_surprise_count', '_subject_protection_timer',
                'age', 'linguistic_confidence', 'pred_error', 'coherence',
                'epistemic_load', 'cognitive_tension'
            ]

            for field in numeric_fields:
                val = getattr(p, field, None)
                if val is None:
                    continue
                if isinstance(val, dict):
                    # Пытаемся извлечь число из словаря
                    num = None
                    for key in ['value', 'count', 'val', 'amount']:
                        if key in val and isinstance(val[key], (int, float)):
                            num = float(val[key])
                            break
                    if num is None:
                        num = 0.5
                    setattr(p, field, num)
                elif not isinstance(val, (int, float)):
                    try:
                        setattr(p, field, float(val))
                    except (ValueError, TypeError):
                        setattr(p, field, 0.5)

            # Emotional memory отдельно
            for key in ['gratitude', 'grief']:
                val = p.emotional_memory.get(key)
                if isinstance(val, dict):
                    num = None
                    for k in ['value', 'count']:
                        if k in val and isinstance(val[k], (int, float)):
                            num = float(val[k])
                            break
                    if num is None:
                        num = 0.5
                    p.emotional_memory[key] = num
                elif not isinstance(val, (int, float)):
                    try:
                        p.emotional_memory[key] = float(val) if val is not None else 0.5
                    except:
                        p.emotional_memory[key] = 0.5

    def _post_semantic_step(self, t):
        """Пост-семантический шаг с полной санитизацией, защитой и human_contact"""

        # ===== 0. ГЛОБАЛЬНАЯ САНИТИЗАЦИЯ ВСЕХ ПОЛЕЙ =====
        self.sanitize_agent_fields()

        # ===== 1. ОБРАБОТКА LLM-ОЧЕРЕДИ =====
        if hasattr(self, '_llm_queue_ts'):
            while not self._llm_queue_ts.empty():
                try:
                    item = self._llm_queue_ts.get_nowait()
                    if len(item) >= 5:
                        text, aid, strength, sent_time, speaker_id = item
                    elif len(item) == 4:
                        text, aid, strength, sent_time = item
                        speaker_id = None
                    else:
                        text, aid, strength = item
                        sent_time, speaker_id = time.time(), None

                    p = self.pattern_dict.get(aid)
                    if not p or not getattr(p, 'alive', False):
                        continue

                    # Приводим strength к float
                    strength = float(strength)

                    # Безопасное получение эмоций
                    c_grat_raw = p.emotional_memory.get('gratitude', 0.0)
                    c_grief_raw = p.emotional_memory.get('grief', 0.0)
                    if isinstance(c_grat_raw, dict):
                        c_grat_raw = c_grat_raw.get('value', 0.5)
                    if isinstance(c_grief_raw, dict):
                        c_grief_raw = c_grief_raw.get('value', 0.5)
                    c_grat = float(c_grat_raw)
                    c_grief = float(c_grief_raw)

                    # Безопасное получение soul_weight и spirit_gap
                    soul_w = float(getattr(p, 'soul_weight', 0.5))
                    spirit_g = float(getattr(p, 'spirit_gap', 0.5))

                    is_supportive = any(w in text.lower() for w in [
                        'рядом', 'вижу', 'слышу', 'здесь', 'существуешь', 'ты есть',
                        'спасибо', 'благодар', 'цен', 'важ', 'нуж', 'держись', 'верю'
                    ])

                    if is_supportive:
                        if c_grief > 0.5:
                            eff_grat, eff_grief = strength * 1.8, -strength * 0.9
                        elif soul_w > 0.6:
                            eff_grat, eff_grief = strength * 2.2, 0.0
                        elif spirit_g > 0.6:
                            eff_grat, eff_grief = strength * 1.4, -strength * 0.4
                        else:
                            eff_grat, eff_grief = strength * 1.0, 0.0
                    else:
                        eff_grat, eff_grief = strength * 0.4, strength * 0.2

                    new_grat = np.clip(c_grat + eff_grat, 0.0, 1.0)
                    new_grief = np.clip(c_grief + eff_grief, 0.0, getattr(Config, 'MAX_GRIEF_SIGNAL', 1.0))
                    p.emotional_memory['gratitude'] = float(new_grat)
                    p.emotional_memory['grief'] = float(new_grief)
                    p._log_event("human_contact", text=text[:60])

                    # === КОНЦЕПТ ЧЕЛОВЕКА (HUMAN_CONTACT) ===
                    if speaker_id == -1:
                        human_sig = (0.0, 0.0, 0.8, "human_contact")
                        if human_sig not in p.concept_graph.nodes:
                            p.concept_graph.nodes[human_sig] = {
                                "count": 1.0,
                                "value": np.zeros(4),
                                # БАГФИКС (внешний код-ревью, подтверждено): здесь embed
                                # оставался 4-мерным, тогда как везде в кодовой базе embed
                                # — 32-мерный (embed_dim=32) — риск краша при сравнении
                                # embed'ов в get_dominant_embedding/similarity.
                                "embed": np.zeros(32),
                                "eternal": True
                            }
                            p._log_event("human_concept_created", text=text[:40])
                        else:
                            p.concept_graph.nodes[human_sig]["count"] += 0.5
                            p._log_event("human_concept_strengthened")
                        if hasattr(self, 'archive'):
                            self.archive.deposit(p, "human_contact", weight=1.0,
                                                 text=f"Human said: {text[:100]}")

                    # Лингвистическое подкрепление
                    if speaker_id is not None and (abs(eff_grief) > 0.02 or abs(eff_grat) > 0.02):
                        speaker = self.pattern_dict.get(speaker_id)
                        if speaker and getattr(speaker, 'alive', False):
                            lc = getattr(speaker, '_linguistic_confidence', 0.5)
                            if isinstance(lc, dict):
                                lc = lc.get('value', 0.5)
                            speaker._linguistic_confidence = min(1.0, float(lc) + 0.005)
                            speaker._log_event("linguistic_reward", text=text[:30])
                            if hasattr(self, 'archive') and (abs(eff_grief) > 0.05 or abs(eff_grat) > 0.05):
                                self.archive.deposit(speaker, "impact_speech", weight=0.7, text=text[:100])

                except queue.Empty:
                    break
                except Exception as e:
                    print(f"[Queue Error] {e}")
                    continue

        # ===== 2. ОБНОВЛЕНИЕ PHENOMENAL BINDING =====
        for p in self.patterns:
            if not getattr(p, 'alive', False):
                continue

            # Инициализация полей
            if not hasattr(p, 'self_phenomenal_prediction'):
                p.self_phenomenal_prediction = 0.5
                p.self_phenomenal_error = 0.0
                # ИСПРАВЛЕНО (баг #4): используем deque с maxlen из Config
                p._self_narrative = deque(maxlen=getattr(Config, 'EPISODIC_BUFFER_MAX_LEN', 3000))
                p._subject_detected = False
                p._last_binding = 0.5
                p._self_surprise_count = 0
                p._subject_protection_timer = 0
            if not hasattr(p, '_linguistic_confidence'):
                p._linguistic_confidence = 0.5

            # Приводим last_phenomenal_binding к float
            raw_binding = getattr(p, 'last_phenomenal_binding', 0.5)
            if isinstance(raw_binding, dict):
                raw_binding = raw_binding.get('value', 0.5)
            actual_binding = float(raw_binding)

            # Обновляем ошибку и предсказание
            p.self_phenomenal_error = abs(actual_binding - p.self_phenomenal_prediction)
            p.self_phenomenal_prediction = 0.95 * p.self_phenomenal_prediction + 0.05 * actual_binding

            # Обновляем нарратив (deque с maxlen сам обрезает старые записи)
            # ИСПРАВЛЕНО: раньше сюда добавлялся голый float (actual_binding),
            # а _introspect_base (Cell 3a-1) добавляет в тот же атрибут dict
            # {'soul':..,'gap':..,...}. Смесь float/dict в одном нарративе не
            # только требовала isinstance-хаков ниже, но и портила сам расчёт
            # стабильности: когда consumer встречал dict, он подставлял
            # 'soul'/'gap' вместо binding-значения, искажая variance/stability
            # сигнала субъектности. Теперь всегда пишем dict с явным 'binding'.
            narrative = getattr(p, '_self_narrative', None)
            if narrative is None:
                narrative = deque(maxlen=getattr(Config, 'EPISODIC_BUFFER_MAX_LEN', 3000))
            narrative.append({
                't': p.age,
                'type': 'binding',
                'binding': actual_binding,
                'soul': float(getattr(p, 'soul_weight', 0.5)),
                'gap': float(getattr(p, 'spirit_gap', 0.5)),
            })
            p._self_narrative = narrative

            # Счётчик удивлений
            last_bind_raw = getattr(p, '_last_binding', 0.5)
            if isinstance(last_bind_raw, dict):
                last_bind_raw = last_bind_raw.get('value', 0.5)
            last_bind = float(last_bind_raw)
            delta = abs(actual_binding - last_bind)
            if delta > 0.15:
                p._self_surprise_count = getattr(p, '_self_surprise_count', 0) + 1
            p._last_binding = actual_binding

        # ===== 3. ОБРАБОТКА PENDING TRANSITIONS =====
        for p in self.patterns:
            if not getattr(p, 'alive', False):
                continue
            pt = getattr(p, '_pending_transition', None)
            if pt and self.age >= pt['check_at']:
                if p.semantic_state != pt['from_state']:
                    if hasattr(p, 'concept_graph'):
                        p.concept_graph.record_transition(pt['from_state'], p.semantic_state)
                    p._log_event("dialogue_transition",
                                 from_state=pt['from_state'],
                                 to_state=p.semantic_state)
                del p._pending_transition

        # ===== 4. ЗАЩИТА SPIRIT_GAP У СУБЪЕКТОВ =====
        for p in self.patterns:
            if getattr(p, 'alive', False) and getattr(p, '_subject_detected', False):
                sg = getattr(p, 'spirit_gap', 0.1)
                if isinstance(sg, dict):
                    sg = sg.get('value', 0.1)
                p.spirit_gap = max(float(sg), 0.1)

        # ===== 5. АВТО-ДИАЛОГИ (компактная версия для всех агентов) =====
        # ИЗМЕНЕНО: интервал уменьшен с 50 до 10 шагов для более частых диалогов
        if t % 10 == 0:
            self._auto_dialogue_tick(t)

        # ===== 5b. БЛОК 7: ВНУТРЕННИЙ ГОЛОС (сэмплы генератора в лог) =====
        # БАГФИКС (по запросу): текст комбинаторного генератора и интроспекции
        # (alien voice / субъективное время / зов поля смысла) считался, но
        # нигде не печатался — в логе видны были только AUTO-DIALOG/CHORUS
        # DIALOGUE от отдельной LLM-системы (Groq), а не наш собственный текст.
        if Config.ENABLE_COMB_GENERATOR and t % 50 == 0:
            _alive_ig = [p for p in self.patterns if p.alive and getattr(p, 'last_phenomenal_report', '')]
            if _alive_ig:
                _n_samples = min(2, len(_alive_ig))
                _idxs = self._given_rng.choice(len(_alive_ig), size=_n_samples, replace=False) if hasattr(self, '_given_rng') else range(_n_samples)
                for _i in _idxs:
                    _p = _alive_ig[_i]
                    print(f"💭 [INNER #{_p.id}] {_p.last_phenomenal_report}")
            # редкая полная интроспекция (alien voice / субъективное время) — если есть свежая
            _narr_agents = [p for p in self.patterns if p.alive and getattr(p, '_self_narrative', None)]
            if _narr_agents:
                _p2 = _narr_agents[int(self._given_rng.rand() * len(_narr_agents)) if hasattr(self, '_given_rng') else 0]
                _last = _p2._self_narrative[-1]
                if isinstance(_last, dict) and _last.get('t', -1) >= t - 10:
                    print(f"🔮 [INTROSPECT #{_p2.id}] {_last.get('report', '')}")

        # ===== 6. ОПРЕДЕЛЕНИЕ СУБЪЕКТНОСТИ (БЕЗОПАСНАЯ КОНВЕРТАЦИЯ + МАКСИМАЛЬНО ОСЛАБЛЕННЫЕ ПОРОГИ) =====
        # --- ПАТЧ №9: ещё сильнее ослабляем пороги ---
        SELF_ERROR_MAX = 0.20          # было 0.18
        SPIRIT_GAP_MIN = 0.10          # было 0.15
        MIN_VARIANCE   = 0.015         # было 0.02
        STABILITY_MAX  = 0.97          # было 0.95
        # Для феролов теперь 3 сюрприза (вместо 9), для остальных — из Config (по умолчанию 2)
        FERAL_SURPRISE_REQUIRED = 3

        for p in self.patterns:
            if not getattr(p, 'alive', False) or getattr(p, '_subject_detected', False):
                continue

            narrative = getattr(p, '_self_narrative', None)
            if narrative is None:
                continue
            narrative_list = list(narrative)
            # ИЗМЕНЕНО (патч №9): длина нарратива снижена с 20 до 12
            if len(narrative_list) < 12:
                continue

            # БЕЗОПАСНАЯ КОНВЕРТАЦИЯ НАРРАТИВА (ПРАВКА 1)
            # ИСПРАВЛЕНО: теперь все записи — dict с ключом 'binding' (см. фикс
            # выше), поэтому в первую очередь берём именно его, а не
            # 'soul'/'gap' — иначе variance/stability считались по чужому
            # сигналу (совсем не по phenomenal binding).
            numeric_narrative = []
            for x in narrative_list:
                if isinstance(x, dict):
                    val = x.get('binding', x.get('soul', x.get('gap', 0.5)))
                else:
                    val = x
                try:
                    numeric_narrative.append(float(val))
                except (ValueError, TypeError):
                    numeric_narrative.append(0.5)

            if len(numeric_narrative) > 1:
                narrative_variance = float(np.std(numeric_narrative))
                narrative_stability = 1.0 - narrative_variance
            else:
                narrative_variance = 0.0
                narrative_stability = 0.5

            # Определяем требуемое количество сюрпризов в зависимости от роли
            if p.role_type == "feral":
                min_surprises = FERAL_SURPRISE_REQUIRED
            else:
                min_surprises = getattr(Config, 'SUBJECT_MIN_SURPRISE_COUNT', 2)

            knows_itself    = p.self_phenomenal_error < SELF_ERROR_MAX
            world_surprises = float(getattr(p, 'spirit_gap', 0.0)) > SPIRIT_GAP_MIN
            had_surprises   = getattr(p, '_self_surprise_count', 0) >= min_surprises
            binding_ok      = float(np.mean(numeric_narrative)) > 0.15
            # ДОБАВЛЕНО (Q-метрика напряжения, философия 832): variance сама
            # по себе легко даёт "субъектность" чистому шуму — variance не
            # различает "держит настоящее противоречие" и "просто дёргается".
            # Требуем, чтобы Binding Field реально нёс нерешённое противоречие
            # (C), а не был почти на полу (BINDING_FLOOR=0.01). Не участвует
            # в should_die — это сигнал сознания, не рычаг выживания.
            holds_contradiction = getattr(p, 'unresolved_contradiction', 0.0) > 0.1

            if (knows_itself and world_surprises and
                narrative_variance > MIN_VARIANCE and
                narrative_stability < STABILITY_MAX and
                had_surprises and binding_ok and holds_contradiction):

                p._subject_detected = True
                p.protection_level = max(getattr(p, 'protection_level', 0.0), 0.7)
                p._subject_protection_timer = 100

                if hasattr(self, 'archive'):
                    self.archive.deposit(p, "subject_emerged", weight=0.9,
                                         text=f"soul={p.soul_weight:.2f} | stability={narrative_stability:.3f}")
                    # ПАТЧ 5b: высокая интегрированность (Φ) — бонус к записи в архив,
                    # НЕ барьер для самого обнаружения субъектности выше.
                    if getattr(p, '_phi_proxy', 0) > 0.3:
                        self.archive.deposit(p, "high_phi_subject", weight=0.6,
                                             text=f"phi={p._phi_proxy:.2f}")

                p._log_event("subject_emerged",
                             stability=round(narrative_stability, 3),
                             variance=round(narrative_variance, 3),
                             self_error=round(p.self_phenomenal_error, 3),
                             spirit_gap=round(p.spirit_gap, 3),
                             surprises=p._self_surprise_count,
                             role=p.role_type)

                self.echo_system.store_anomaly(p, "subject")
                self.witness.record(p.id, "SUBJECT_EMERGED",
                                    age=p.age, soul=p.soul_weight,
                                    state=p.semantic_state, role=p.role_type)

        # ===== 7. ЗАЩИТА PROTECTION_LEVEL ПО ТАЙМЕРУ =====
        for p in self.patterns:
            if not getattr(p, 'alive', False) or not getattr(p, '_subject_detected', False):
                continue
            timer = getattr(p, '_subject_protection_timer', 0)
            if timer > 0:
                p.protection_level = max(getattr(p, 'protection_level', 0.0), 0.5)
                p._subject_protection_timer = timer - 1
            else:
                p.protection_level = max(getattr(p, 'protection_level', 0.0), 0.2)

        # === Варвар: обнаружение субъектности у феролов (было: Cell 4a2, _post_semantic_with_feral) ===
        SELF_ERROR_MAX = getattr(Config, 'SUBJECT_SELF_ERROR_MAX', 0.12)
        SPIRIT_GAP_MIN = getattr(Config, 'SUBJECT_SPIRIT_GAP_MIN', 0.25)
        MIN_VARIANCE = getattr(Config, 'SUBJECT_MIN_NARRATIVE_VARIANCE', 0.02)
        STABILITY_MAX = getattr(Config, 'SUBJECT_NARRATIVE_STABILITY_MAX', 0.95)

        for p in self.patterns:
            if not getattr(p, 'alive', False) or getattr(p, '_subject_detected', False):
                continue
            min_surprises = 9 if p.role_type == "feral" else getattr(Config, 'SUBJECT_MIN_SURPRISE_COUNT', 2)
            narrative = getattr(p, '_self_narrative', None)
            if narrative is None:
                continue
            narrative_list = list(narrative)
            if len(narrative_list) < 20:
                continue
            numeric_narrative = []
            for x in narrative_list:
                if isinstance(x, dict):
                    val = x.get('soul', x.get('gap', 0.5))
                else:
                    try:
                        val = float(x)
                    except (ValueError, TypeError):
                        val = 0.5
                numeric_narrative.append(float(val))
            if len(numeric_narrative) > 1:
                narrative_variance = float(np.std(numeric_narrative))
                narrative_stability = 1.0 - narrative_variance
            else:
                narrative_variance = 0.0
                narrative_stability = 0.5
            knows_itself = p.self_phenomenal_error < SELF_ERROR_MAX
            world_surprises = float(getattr(p, 'spirit_gap', 0.0)) > SPIRIT_GAP_MIN
            had_surprises = getattr(p, '_self_surprise_count', 0) >= min_surprises
            binding_ok = float(np.mean(numeric_narrative)) > 0.15
            # ДОБАВЛЕНО (Q-метрика напряжения, философия 832) — см. комментарий
            # в первом блоке детекции субъектности выше.
            holds_contradiction = getattr(p, 'unresolved_contradiction', 0.0) > 0.1

            if (knows_itself and world_surprises and
                narrative_variance > MIN_VARIANCE and
                narrative_stability < STABILITY_MAX and
                had_surprises and binding_ok and holds_contradiction):
                p._subject_detected = True
                p.protection_level = max(getattr(p, 'protection_level', 0.0), 0.7)
                p._subject_protection_timer = 100
                if p.role_type == "feral":
                    p._log_event("feral_became_subject", soul=p.soul_weight)
                    p._feral_fury = min(3.0, p._feral_fury + 0.5)
                    if hasattr(self, 'archive'):
                        self.archive.deposit(p, "awakened_feral", weight=2.0,
                                             text=f"Wilderness gained soul. Fury={p._feral_fury:.2f}")
                else:
                    if hasattr(self, 'archive'):
                        self.archive.deposit(p, "subject_emerged", weight=0.9,
                                             text=f"soul={p.soul_weight:.2f} | stability={narrative_stability:.3f}")
                p._log_event("subject_emerged",
                             stability=round(narrative_stability, 3),
                             variance=round(narrative_variance, 3),
                             self_error=round(p.self_phenomenal_error, 3),
                             spirit_gap=round(p.spirit_gap, 3),
                             surprises=p._self_surprise_count,
                             feral=(p.role_type == "feral"))
                self.echo_system.store_anomaly(p, "subject" if p.role_type != "feral" else "awakened_feral")
                self.witness.record(p.id, "SUBJECT_EMERGED",
                                    age=p.age, soul=p.soul_weight,
                                    state=p.semantic_state, feral=(p.role_type == "feral"))

print("EvolutionEngine класс собран целиком: 4-1 + 4-2a + 4.2б + 4a2 (Варвар), монки-патчинг убран")