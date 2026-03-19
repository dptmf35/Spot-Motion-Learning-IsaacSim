# 보스턴 다이나믹스 Spot 로봇용 보행 데이터 수집 시스템 (Isaac Sim 5.0)

**[English version below / 아래에 영문 버전이 있습니다]**

---

## 목차
1. [시스템 개요](#시스템-개요)
2. [설치 및 실행](#설치-및-실행)
3. [아키텍처 및 데이터 흐름](#아키텍처-및-데이터-흐름)
4. [핵심 컴포넌트](#핵심-컴포넌트)
5. [초기화 시퀀스](#초기화-시퀀스)
6. [데이터 스키마](#데이터-스키마)
7. [중요한 설계 결정 및 주의사항](#중요한-설계-결정-및-주의사항)
8. [대시보드 사용법](#대시보드-사용법)
9. [문제 해결](#문제-해결)

---

## 시스템 개요

이 시스템은 **Isaac Sim 5.0 독립 실행형** 환경에서 Boston Dynamics Spot 로봇의 보행 데이터를 자동으로 수집합니다. 수집된 데이터(관절 위치/속도, 접촉력, ZMP, 발 위치 등)는 지도 학습 기반의 역기구학(IK) 모션 학습에 사용됩니다.

**핵심 특징:**
- 완전 자동화된 에피소드 기반 수집 (user input 불필요)
- 이중 정책 스택: 내비게이션 정책(5 Hz) → 보행 정책(50 Hz)
- 자동 웨이포인트 생성 및 목표 도달 감지
- 고급 접촉 센서 통합 (Isaac Sim 5.0 native API)
- 실시간 웹 대시보드로 수집 과정 모니터링
- 확장 가능한 HDF5 에피소드 저장소

**원하는 출력:**
- HDF5 파일: `data/gait_recordings/gait_session_YYYYMMDD_HHMMSS.h5`
- 각 에피소드당 ~500-1000 타임스텝 (60초 에피소드 기준)
- 필드: 관절 상태, 속도 명령어, 접촉력, ZMP, 발 위치 등

---

## 설치 및 실행

### 사전 요구사항

- Isaac Sim 5.0 독립 실행형 버전이 설치되어 있어야 합니다
- Python 패키지: `h5py`, `fastapi`, `uvicorn`, `numpy`, `torch`

### 기본 실행

```bash
cd /workspaces/IsaacRobotics

# Isaac Sim Python 인터프리터를 통해 실행
~/isaac-sim-standalone-5.0.0-linux-x86_64/python.sh \
  applications/spot_sensors_nav_data_collection.py \
  --num-episodes 100 \
  --episode-duration 60.0 \
  --dashboard-port 8001
```

### 주요 CLI 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--num-episodes N` | 10 | 수집할 에피소드 개수 |
| `--episode-duration SEC` | 60.0 | 에피소드당 최대 지속시간 (초) |
| `--spawn-x X` | -8.0 | 로봇 초기 스폰 X 좌표 |
| `--spawn-y Y` | 4.0 | 로봇 초기 스폰 Y 좌표 |
| `--spawn-yaw YAW` | 0.0 | 로봇 초기 스폰 방향 (라디안) |
| `--arena-x-min/max` | -35.0 / 4.0 | 웨이포인트 샘플링 X 범위 |
| `--arena-y-min/max` | -25.0 / 28.0 | 웨이포인트 샘플링 Y 범위 |
| `--min-goal-dist M` | 2.0 | 목표 최소 거리 (미터) |
| `--max-goal-dist M` | 15.0 | 목표 최대 거리 (미터) |
| `--pos-thresh T` | 0.5 | 위치 도달 임계값 (미터) |
| `--yaw-thresh T` | 0.4 | 회전 도달 임계값 (라디안) |
| `--dashboard-port PORT` | 8001 | 웹 대시보드 포트 |
| `--no-auto-start` | - | 대시보드에서 수동 시작 (자동 시작 안 함) |

### 예제: 자동 수집 200 에피소드 (warehouse 환경)

```bash
~/isaac-sim-standalone-5.0.0-linux-x86_64/python.sh \
  applications/spot_sensors_nav_data_collection.py \
  --num-episodes 200 \
  --episode-duration 60.0 \
  --arena-x-min -35 --arena-x-max 4 \
  --arena-y-min -25 --arena-y-max 28
```

### 웹 대시보드 접근

브라우저에서 다음 주소로 접속:
```
http://localhost:8001
```

대시보드는 컬렉션 상태, 수집된 에피소드 수, 로봇 궤적, 관절 각도, 접촉 상태를 실시간으로 표시합니다.

---

## 아키텍처 및 데이터 흐름

### 이중 정책 스택

보행 데이터 수집은 계층적 정책 구조를 사용합니다:

```
목표 위치 (x, y, yaw) - 경계 내 무작위 샘플링
    ↓
RandomWaypointGenerator
    ↓ (100 physics steps마다 = 0.2초 = 5 Hz decimation)
SpotNavigationPolicy (정책 평가)
    obs: [lin_vel(3), gravity(3), pose_command(3 or 4)]
    → 속도 명령어: [vx, vy, wz]
    ↓
SpotFlatTerrainPolicy (보행 제어)
    obs: [lin_vel(3), ang_vel(3), gravity(3), command(3), joint_pos(12),
          joint_vel(12), prev_action(12)] = 48-dim
    → 관절 위치 명령어 (12 DoF)
    ↓ (매 physics step마다 적용)
Isaac Sim 물리 엔진 (500 Hz, dt=0.002초)
    → 로봇 신체와 환경의 상호작용 시뮬레이션
    ↓
ContactSensorBridge (5 Hz로 접촉력 수집)
    → 발 접촉력, ZMP, 발 위치
    ↓
GaitRecorder (버퍼에 타임스텝 저장)
```

### 에피소드 라이프사이클

```
start_collection() 호출 (자동 시작 또는 대시보드)
    ↓
로봇을 고정된 스폰 위치에서 초기화
    ↓
경계 범위 내에서 무작위 목표 샘플링
    ↓
내비게이션 정책 → 보행 정책 → 로봇이 목표로 이동
    ↓
목표 도달? → 새 목표 샘플링, 계속 이동
    ↓
에피소드 시간 초과 (예: 60초)?
    ↓
버퍼된 보행 데이터 → HDF5 파일에 저장
    ↓
로봇을 고정 스폰 위치로 텔레포트 (world.reset() 사용 안 함)
    ↓
다음 에피소드 시작
    ↓
N개 에피소드 완료? → 대기 (시뮬레이션 계속 실행)
    ↓
대시보드: "수집 시작" → 스폰 위치에서 다시 반복
```

### 수집 재시작 동작

- **첫 시작**: 시뮬레이션 이미 초기화됨, `world.reset()` 호출 안 함
- **후속 시작** (중지 후): `world.reset(True)` 호출로 시뮬레이션을 초기 상태로 재설정
- **에피소드 종료 후**: `set_world_pose()` + `set_joint_positions()`로 로봇을 스폰 위치로 텔레포트 (world.reset() 불필요)
- **모든 에피소드 완료 후**: 시뮬레이션이 계속 실행되고 로봇은 유휴 상태, 대시보드에서 새 수집 트리거 가능

---

## 핵심 컴포넌트

### 1. SpotGaitDataCollector (메인 클래스)

**위치**: `applications/spot_sensors_nav_data_collection.py`

단일 로봇 (`/World/Spot`) 관리 및 보행 데이터 수집을 담당합니다.

**주요 메서드:**

| 메서드 | 설명 |
|--------|------|
| `__init__()` | 로봇 USD 프림, 정책, 내비게이션 매니저 초기화 |
| `setup()` | ContactSensor 프림 생성, 대시보드 서버 시작 |
| `run()` | 메인 루프: 시뮬레이션 스텝 → 정책 평가 → 데이터 수집 |
| `_on_physics_step()` | 매 physics step (500 Hz)마다 호출, 보행 정책 쿼리 및 상태 수집 |
| `_flush_episode()` | 버퍼된 보행 데이터를 HDF5에 저장 |
| `_reset_to_spawn()` | 로봇을 고정 스폰 위치로 텔레포트 및 초기화 |

**핵심 상태 변수:**

```python
self._nav_decimation = 100        # 100 physics steps마다 내비게이션 정책 평가 (5 Hz)
self._spawn_pos = np.array([...]) # 고정 스폰 위치 (에피소드마다 복귀)
self._step_buf = []               # 현재 에피소드의 타임스텝 버퍼
self._current_wp = None           # 현재 목표 웨이포인트 (x, y, yaw)
self._base_cmd = np.zeros(3)      # 현재 속도 명령어 [vx, vy, wz]
```

### 2. AutoCollectionManager (스레드 안전 상태 머신)

**위치**: `applications/spot_sensors_nav_data_collection.py`

Isaac Sim physics 스레드와 FastAPI 스레드 간의 에피소드 라이프사이클을 관리합니다.

**공개 API:**

```python
start_collection(config=None)  # 수집 시작, 선택적 설정 업데이트
stop_collection()              # 수집 중지
consume_world_reset_request()  # 일회성 world.reset() 플래그 확인
get_status()                   # 수집 상태 조회
get_config()                   # 현재 설정 조회
update_config(config)          # 설정 업데이트
is_collecting()                # 수집 중인지 확인
is_complete()                  # 모든 에피소드 완료되었는지 확인
```

**반환 상태:**
```python
{
    "active": bool,                 # 현재 수집 중?
    "current_episode": int,         # 현재 에피소드 번호
    "total_episodes": int,          # 총 에피소드 개수
    "episode_duration": float,      # 현재 에피소드 경과 시간 (초)
    "current_waypoint": [x, y, yaw],  # 현재 목표 웨이포인트
    "goals_reached": int,           # 도달한 목표 개수
}
```

### 3. RandomWaypointGenerator

**위치**: `applications/spot_sensors_nav_data_collection.py`

경계 범위 내에서 무작위 (x, y, yaw) 목표를 샘플링합니다.

**특징:**
- 최소/최대 거리 제약 조건 (로봇이 너무 가깝거나 멀지 않은 목표 선택)
- 유효한 점을 찾지 못하면 50회 재시도 후 순수 무작위 샘플링으로 폴백
- 회전각(yaw)은 항상 [-π, π] 범위에서 무작위 선택

### 4. ContactSensorBridge (접촉력 수집)

**위치**: `dashboard/backend/contact_sensor_bridge.py`

Isaac Sim 5.0의 네이티브 ContactSensor API를 사용하여 발 접촉력을 수집합니다.

#### 중요: 라이프사이클 순서

```python
# Step 1: 센서 프림 생성 (world.reset() 이전)
bridge = ContactSensorBridge("/World/Spot")
bridge.pre_reset_setup()  # PhysxContactReportAPI 적용, 센서 프림 생성

# Step 2: 센서 프림 활성화 (world.reset() 호출)
world.reset()  # ← 반드시 pre_reset_setup() 다음에 호출

# Step 3: 센서 초기화 (world.reset() 이후)
bridge.post_reset_setup(physics_dt=0.002)

# Step 4: 매 physics step마다 업데이트
bridge.update(dt)
forces = bridge.get_contact_forces()  # (4, 3) [Fx, Fy, Fz]
```

#### API

```python
# 접촉력 조회 (Newton)
get_contact_forces()  # → (4, 3) float32 FL/FR/HL/HR 순서

# 공중 시간 조회 (초)
get_air_times()       # → (4,) float32

# 발이 접촉 중인지 확인
get_foot_in_contact()  # → (4,) bool

# ZMP 계산 (압력 중심 방법)
zmp = compute_zmp(foot_positions=(4,3), contact_forces=(4,3))
# → (2,) [x, y] 또는 [nan, nan] (접촉 없을 때)
```

#### 왜 ContactSensor를 사용하나?

Isaac Sim 5.0에서 `PhysX.create_rigid_contact_view` API는 손상되어 있습니다:
- 항상 0을 반환
- `filter_patterns_expr` 키워드 인수가 잘못됨
- 독립 실행형 앱에서 작동하지 않음

**ContactSensor**는 Isaac Sim 5.0의 공식 권장 API이며 독립 실행형 앱에서 제대로 작동합니다.

#### 발 순서

| 인덱스 | 발 | USD 프림 이름 |
|--------|-----|--------------|
| 0 | Front Left (FL) | `fl_foot` |
| 1 | Front Right (FR) | `fr_foot` |
| 2 | Hind Left (HL) | `hl_foot` |
| 3 | Hind Right (HR) | `hr_foot` |

### 5. GaitRecorder (HDF5 저장소)

**위치**: `dashboard/backend/gait_recorder.py`

보행 데이터를 HDF5 형식으로 에피소드 단위로 저장합니다.

**파일 구조:**

```
data/gait_recordings/gait_session_20260319_154322.h5
├── attributes:
│   ├── created_at: "20260319_154322"
│   ├── robot_type: "SpotFlat"
│   ├── obs_dim: 48
│   └── action_dim: 12
└── episode_0000/
    ├── timestamps (N,)
    ├── obs (N, 48)
    ├── actions (N, 12)
    ├── commands (N, 3)
    ├── body_pose (N, 7)
    ├── body_lin_vel (N, 3)
    ├── body_ang_vel (N, 3)
    ├── com_position (N, 3)
    ├── zmp_position (N, 2)
    ├── foot_positions (N, 4, 3)
    ├── foot_contact_forces (N, 4, 3)
    ├── foot_air_time (N, 4)
    ├── head_position (N, 3)
    ├── joint_pos (N, 12)
    ├── joint_vel (N, 12)
    └── attributes:
        ├── episode_num: 0
        ├── start_time: 1710861822.5
        ├── end_time: 1710861882.5
        ├── duration_sec: 60.0
        ├── n_steps: 300
        ├── waypoints: "[[-15.2, 8.3, 0.5], ...]"
        └── arena_bounds: "{...}"
```

**API:**

```python
recorder = GaitRecorder(save_dir="data/gait_recordings")

# 새 세션 시작
session_file = recorder.start_session()  # HDF5 파일 생성

# 에피소드 시작
episode_idx = recorder.start_episode(arena_bounds={...})

# 매 타임스텝마다 데이터 추가
recorder.add_step({
    "timestamps": time.time(),
    "obs": np.zeros(48),
    "actions": np.zeros(12),
    # ... 다른 필드들
})

# 웨이포인트 기록
recorder.record_waypoint((x, y, yaw))

# 에피소드 종료 (버퍼를 HDF5에 저장)
metadata = recorder.stop_episode()

# 여러 에피소드를 한 세션에 저장
episode_idx = recorder.start_episode(...)
# ... 데이터 추가
recorder.stop_episode()

# 세션 종료
recorder.close()
```

### 6. 대시보드 백엔드 (FastAPI 서버)

**위치**: `dashboard/backend/gait_main.py`

웹 기반 모니터링 및 제어를 위한 REST API와 WebSocket 엔드포인트를 제공합니다.

**REST 엔드포인트:**

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/api/status` | 연결 상태, 녹화 상태, 수집 진행 상황 |
| POST | `/api/collection/start` | 수집 시작 (본체: `{num_episodes, episode_duration}`) |
| POST | `/api/collection/stop` | 수집 중지 |
| GET | `/api/collection/config` | 현재 수집 설정 조회 |
| PUT | `/api/collection/config` | 수집 설정 업데이트 |
| GET | `/api/episodes` | 녹화된 에피소드 목록 |

**WebSocket:**

```
ws://localhost:8001/ws
```

10 Hz로 다음 상태를 푸시:
- 현재 보행 상태 (관절 각도, 접촉력, 발 위치 등)
- 수집 상태 (현재 에피소드, 목표 등)
- 센서 데이터

### 7. 대시보드 프론트엔드 (React)

**위치**: `dashboard/frontend/src/tabs/GaitDataCollection.tsx`

웹 기반 UI로 실시간 모니터링과 수집 제어를 제공합니다.

**기능:**
- 수집 상태 표시 (오프라인/온라인, 활성/비활성)
- 에피소드 진행 바
- 설정 입력: 에피소드 개수, 에피소드 지속시간
- 2D 궤적 차트 (CoM 경로 + ZMP 궤적)
- 관절 각도 차트 (Hip Ab/Ad, Hip Flexion, Knee)
- 발 접촉 상태 표시기 (FL/FR/HL/HR 버튼)
- 발 접촉력 막대 그래프
- 에피소드 기록 목록

---

## 초기화 시퀀스

올바른 초기화 순서는 **매우 중요**합니다. 특히 ContactSensor는 `world.reset()` 전에 생성되어야 올바르게 활성화됩니다.

```
main() 시작
    ↓
SpotGaitDataCollector(...) 생성
    - 로봇 USD 프림 생성 (/World/Spot)
    - 내비게이션 정책 로드
    - 보행 정책 로드
    ↓
simulation_app.update()
    ↓
collector._world.reset()  # 첫 번째 reset
    - warehouse USD 로드
    - 로봇 USD 파일 로드
    ↓
simulation_app.update()
    ↓
collector.setup()
    - bridge.pre_reset_setup() 호출
        └─ PhysxContactReportAPI를 rigid bodies에 적용
        └─ USD stage 순회로 발 프림 발견 (fl_foot, fr_foot, hl_foot, hr_foot)
        └─ ContactSensor 프림 생성
    - physics callback 등록 (_on_physics_step)
    - FastAPI 대시보드 서버 시작
    ↓
collector._world.reset()  # 두 번째 reset
    - ContactSensor 프림 활성화 (이제 데이터 보고 시작)
    ↓
simulation_app.update()
    ↓
collector.run()  # 메인 루프 시작
    - while not exit:
        ├─ world.step(render=True)
        ├─ _on_physics_step() 콜백 (500 Hz)
        ├─ 내비게이션 정책 평가 (100 step마다)
        ├─ 보행 데이터 수집
        ├─ 에피소드 종료 시 HDF5에 저장
        └─ 로봇을 스폰 위치로 텔레포트
```

**왜 두 개의 world.reset() 호출이 필요한가?**

1. **첫 번째 reset**: USD stage를 로드합니다 (warehouse, Spot).
2. **setup() 호출**: ContactSensor 프림이 USD stage에 추가됩니다.
3. **두 번째 reset**: 이제 존재하는 ContactSensor 프림이 활성화되고 접촉력 데이터를 보고하기 시작합니다.

첫 번째 reset 없이 setup()을 호출하면 stage가 아직 로드되지 않았기 때문에 발 프림을 찾을 수 없습니다. 두 번째 reset 없이 계속하면 ContactSensor가 활성화되지 않아 항상 0을 반환합니다.

---

## 데이터 스키마

### 관찰 벡터 (obs) - 48차원

보행 정책의 입력:

```
[lin_vel(3), ang_vel(3), gravity(3), command(3), joint_pos(12), joint_vel(12), prev_action(12)]
  0-2        3-5        6-8        9-11       12-23        24-35          36-47
```

| 필드 | 차원 | 설명 |
|------|------|------|
| `lin_vel` | 3 | 선형 속도 (세계 좌표) |
| `ang_vel` | 3 | 각속도 (세계 좌표) |
| `gravity` | 3 | 중력 (로봇 본체 좌표, 기울기 감지용) |
| `command` | 3 | 속도 명령어 [vx, vy, wz] (내비게이션 정책의 출력) |
| `joint_pos` | 12 | 관절 위치 (4개 다리 × 3 DoF) |
| `joint_vel` | 12 | 관절 속도 |
| `prev_action` | 12 | 이전 타임스텝의 동작 |

### 동작 벡터 (actions) - 12차원

보행 정책의 출력 (관절 위치 델타):

```
[FL_hip_aa, FL_hip_fe, FL_knee,
 FR_hip_aa, FR_hip_fe, FR_knee,
 HL_hip_aa, HL_hip_fe, HL_knee,
 HR_hip_aa, HR_hip_fe, HR_knee]
```

각 값은 기본 자세에서의 이동량입니다 (0.2 스케일).

### HDF5 데이터셋 구조

각 에피소드 그룹 내의 데이터셋:

| 데이터셋 | 형태 | 타입 | 설명 |
|---------|------|------|------|
| `timestamps` | (N,) | float64 | Unix 타임스탬프 |
| `obs` | (N, 48) | float32 | 정책 관찰 벡터 |
| `actions` | (N, 12) | float32 | 관절 위치 명령어 |
| `commands` | (N, 3) | float32 | 속도 명령어 [vx, vy, wz] |
| `body_pose` | (N, 7) | float32 | 위치(3) + 사원수(4) [w,x,y,z] |
| `body_lin_vel` | (N, 3) | float32 | 선형 속도 (세계 좌표) |
| `body_ang_vel` | (N, 3) | float32 | 각속도 (세계 좌표) |
| `com_position` | (N, 3) | float32 | 질량 중심 위치 (≈ 로봇 본체 위치) |
| `zmp_position` | (N, 2) | float32 | Zero Moment Point [x, y] |
| `foot_positions` | (N, 4, 3) | float32 | 발 위치 (세계 좌표) FL/FR/HL/HR |
| `foot_contact_forces` | (N, 4, 3) | float32 | 지면 반력 [Fx, Fy, Fz] FL/FR/HL/HR |
| `foot_air_time` | (N, 4) | float32 | 각 발의 공중 지속 시간 (초) |
| `head_position` | (N, 3) | float32 | 머리 위치 (별도 머리 링크 없음) |
| `joint_pos` | (N, 12) | float32 | 관절 위치 |
| `joint_vel` | (N, 12) | float32 | 관절 속도 |

### 에피소드 메타데이터 (HDF5 속성)

```python
episode_group.attrs = {
    "episode_num": 0,                    # 에피소드 번호
    "start_time": 1710861822.5,          # 시작 Unix 타임스탬프
    "end_time": 1710861882.5,            # 종료 Unix 타임스탐프
    "duration_sec": 60.0,                # 에피소드 지속시간
    "n_steps": 300,                      # 타임스텝 수
    "waypoints": "[[-15.2, 8.3, 0.5], ...]",  # 방문한 웨이포인트 (JSON)
    "arena_bounds": "{...}",             # 수집 경계 (JSON)
}
```

---

## 중요한 설계 결정 및 주의사항

### 1. 두 번의 world.reset() 호출

**문제**: ContactSensor 프림이 USD stage에 없으면 활성화되지 않습니다.

**해결책**:
```python
collector._world.reset()  # 첫 번째: USD 로드
collector.setup()         # ContactSensor 프림 생성
collector._world.reset()  # 두 번째: 프림 활성화
```

**결과**: 두 번째 reset 이후에야 접촉력 데이터가 0이 아닌 값을 반환합니다.

### 2. ContactSensor vs PhysX Tensor API

**왜 ContactSensor를 사용하는가?**

```python
# ❌ 이것은 Isaac Sim 5.0에서 작동하지 않음
contact_view = create_rigid_contact_view(...)  # 항상 0 반환
```

```python
# ✅ 올바른 방법 (Isaac Sim 5.0)
bridge = ContactSensorBridge(...)
bridge.pre_reset_setup()   # ContactSensor 프림 생성
world.reset()              # 활성화
bridge.update(dt)          # 매 step에 업데이트
forces = bridge.get_contact_forces()  # 0이 아닌 값 반환
```

### 3. 에피소드 종료 후 world.reset() 사용 안 함

**최적화**: 각 에피소드 후 전체 시뮬레이션을 재설정하지 않습니다.

```python
# ❌ 느림 (불필요한 리소스 사용)
world.reset()
robot.initialize()

# ✅ 빠름 (개별 상태만 재설정)
robot.set_world_pose(spawn_pos, spawn_quat)
robot.set_joint_positions(default_pos)
robot.set_joint_velocities(zeros)
```

50 에피소드 × 60초 = 3000초 = 약 50분
- `world.reset()` 사용: 매번 5-10초 지연 → 총 250-500초 추가
- 텔레포트만 사용: 무시할 수 있는 지연

### 4. 내비게이션 정책 관찰 차원 자동 감지

내비게이션 정책은 3-dim 또는 4-dim 위치 명령어를 수용할 수 있습니다:

```python
pose_dim = nav_policy.obs_dim - 6  # 6 = lin_vel(3) + gravity(3)

if pose_dim == 4:
    # [dx, dy, dz, heading_error]
    pose_command = np.array([pos_cmd[0], pos_cmd[1], pos_cmd[2], heading])
else:
    # [dx, dy, heading_error]
    pose_command = np.array([pos_cmd[0], pos_cmd[1], heading])
```

### 5. 회전 좌표계 변환

위치 명령어는 **로봇 본체 좌표**로 변환됩니다 (IsaacLab 컨벤션):

```python
# 세계 좌표의 목표 위치
target_world = [goal_x, goal_y, robot_z]
target_vec = target_world - robot_pos

# 로봇 yaw 회전의 역행렬
R_yaw_inv = [[cos(yaw), sin(yaw), 0],
             [-sin(yaw), cos(yaw), 0],
             [0, 0, 1]]

# 로봇 본체 좌표로 변환
pos_command_body = R_yaw_inv @ target_vec
```

### 6. ZMP 계산

Zero Moment Point는 접촉 압력 분포의 중심으로 계산됩니다:

```python
def compute_zmp(foot_positions, contact_forces, threshold=1.0):
    """
    foot_positions: (4, 3) - FL/FR/HL/HR
    contact_forces: (4, 3) - [Fx, Fy, Fz]
    threshold: 최소 수직력 (N) - 아래면 무시
    """
    fz = contact_forces[:, 2]
    mask = fz > threshold
    if not mask.any():
        return [nan, nan]

    fz_v = fz[mask]
    total = fz_v.sum()
    zmp_x = (fz_v * foot_positions[mask, 0]).sum() / total
    zmp_y = (fz_v * foot_positions[mask, 1]).sum() / total
    return [zmp_x, zmp_y]
```

### 7. 웨이포인트 샘플링 제약

목표는 다음 조건을 만족해야 합니다:

```python
min_distance <= distance(robot, goal) <= max_distance
arena_bounds["x_min"] <= goal_x <= arena_bounds["x_max"]
arena_bounds["y_min"] <= goal_y <= arena_bounds["y_max"]
```

50회 재시도 후에도 유효한 점을 찾지 못하면 제약 없이 무작위 샘플링합니다.

---

## 대시보드 사용법

### 웹 인터페이스 접속

```
http://localhost:8001
```

### GaitDataCollection 탭

#### 상태 표시기

- **Offline/Connected**: 시뮬레이션 연결 상태
- **Collecting badge**: 현재 수집 중인지 표시

#### 수집 제어

1. **에피소드 개수** 입력
2. **에피소드 지속시간** (초) 입력
3. **Start Collection** 클릭
4. 진행 상황 모니터링
5. **Stop Collection** 클릭 (선택사항)

#### 실시간 시각화

1. **궤적 차트**: CoM 경로 (파란색) + ZMP 궤적 (빨간색)
2. **관절 각도 차트**: 선택된 관절의 시간대별 각도
3. **발 접촉 상태**: FL/FR/HL/HR의 on/off 상태
4. **접촉력 막대**: 각 발의 수직력 (N)

#### 에피소드 기록

완료된 에피소드 목록:
- 에피소드 번호
- 지속시간
- 도달한 목표 개수
- 저장된 파일

---

## 문제 해결

### 접촉력이 항상 0인 경우

**원인**: ContactSensor가 활성화되지 않음

**해결책**:
```python
# 1. pre_reset_setup() 호출 확인
bridge.pre_reset_setup()

# 2. world.reset() 호출 확인
world.reset()

# 3. post_reset_setup() 호출 확인
bridge.post_reset_setup()

# 4. 로그 확인
# "[ContactBridge] Found foot prims (4): [...]" 메시지 확인
```

### 발 프림을 찾을 수 없음

**원인**: USD stage에 발 프림이 없거나 이름이 다름

**해결책**:
```bash
# Isaac Sim 노드 그래프에서 발 프림 확인
# 다음 이름을 찾으세요: fl_foot, fr_foot, hl_foot, hr_foot

# spot_sensors.usd 파일 확인
grep -i "foot" assets/spot_sensors.usd
```

### 로봇이 움직이지 않음

**원인**: 정책이 로드되지 않았거나 관찰이 잘못됨

**해결책**:
```bash
# 정책 파일 확인
ls -la policies/spot_flat/models/policy.pt
ls -la policies/spot_nav/models/policy.pt

# 로그에서 정책 로드 메시지 확인
# "[GaitCollector] Nav policy loaded. obs_dim=..."
```

### 웹 대시보드가 연결되지 않음

**원인**: FastAPI 서버가 시작되지 않았거나 포트가 사용 중

**해결책**:
```bash
# 포트 확인
lsof -i :8001

# 다른 포트로 지정
python.sh applications/spot_sensors_nav_data_collection.py --dashboard-port 8002

# 브라우저에서 다시 접속
http://localhost:8002
```

### 에피소드 데이터가 저장되지 않음

**원인**: `data/gait_recordings/` 디렉토리 권한 문제

**해결책**:
```bash
mkdir -p data/gait_recordings
chmod 755 data/gait_recordings

# 파일이 생성되는지 확인
ls -la data/gait_recordings/
```

### 메모리 누수 또는 느려지는 성능

**원인**: 큰 버퍼가 자동으로 정리되지 않음

**해결책**:
```python
# 에피소드 종료 후 버퍼 명시적 정리 확인
self._step_buf = []

# HDF5 파일이 닫혔는지 확인
recorder.close()
```

---

## 추가 리소스

### 데이터 분석 예제 (Python)

```python
import h5py
import numpy as np

# HDF5 파일 열기
with h5py.File('data/gait_recordings/gait_session_20260319_154322.h5', 'r') as f:
    # 에피소드 목록
    episodes = [k for k in f.keys() if k.startswith('episode_')]

    for ep_name in episodes[:5]:  # 처음 5개 에피소드
        ep = f[ep_name]

        # 메타데이터
        print(f"\n{ep_name}:")
        print(f"  Steps: {len(ep['timestamps'])}")
        print(f"  Duration: {ep.attrs['duration_sec']:.1f}s")
        print(f"  Goals reached: {len(ep.attrs['waypoints'])}")

        # 관절 각도 범위
        joint_pos = ep['joint_pos'][:]
        print(f"  Joint pos range: [{joint_pos.min():.3f}, {joint_pos.max():.3f}]")

        # 최대 접촉력
        forces = ep['foot_contact_forces'][:]
        max_force = forces[:, :, 2].max()
        print(f"  Max contact force: {max_force:.1f} N")
```

### 데이터 검증

```python
def validate_episode(h5_file, episode_idx):
    """에피소드 데이터 검증"""
    with h5py.File(h5_file, 'r') as f:
        ep = f[f'episode_{episode_idx:04d}']
        n_steps = len(ep['timestamps'])

        issues = []

        # 모든 데이터셋 길이 확인
        for key in ep.keys():
            if len(ep[key]) != n_steps:
                issues.append(f"{key}: shape {ep[key].shape} != {n_steps}")

        # NaN 확인
        for key in ['zmp_position', 'foot_contact_forces']:
            if np.isnan(ep[key][:]).all():
                issues.append(f"{key}: all NaN")

        # 타임스탬프 순서 확인
        ts = ep['timestamps'][:]
        if not np.all(np.diff(ts) > 0):
            issues.append("Timestamps not monotonic")

        if issues:
            print(f"⚠ Episode {episode_idx} issues:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print(f"✓ Episode {episode_idx} valid")

        return len(issues) == 0
```

---

<br><br>

# Autonomous Gait Data Collection System for Boston Dynamics Spot (Isaac Sim 5.0)

---

## Table of Contents
1. [System Overview](#system-overview)
2. [Installation and Running](#installation-and-running)
3. [Architecture and Data Flow](#architecture-and-data-flow)
4. [Core Components](#core-components)
5. [Initialization Sequence](#initialization-sequence)
6. [Data Schema](#data-schema)
7. [Critical Design Decisions and Gotchas](#critical-design-decisions-and-gotchas)
8. [Dashboard Usage](#dashboard-usage)
9. [Troubleshooting](#troubleshooting)

---

## System Overview

This system automatically collects gait data for the Boston Dynamics Spot robot running in **Isaac Sim 5.0 standalone** mode. The collected data (joint positions/velocities, contact forces, ZMP, foot positions, etc.) is used for supervised learning-based inverse kinematics (IK) motion learning.

**Key Features:**
- Fully automated, episode-based data collection (no user input required)
- Dual policy stack: navigation policy (5 Hz) → locomotion policy (50 Hz)
- Automatic waypoint generation and goal-reaching detection
- Advanced contact sensor integration (Isaac Sim 5.0 native API)
- Real-time web dashboard for monitoring collection progress
- Extensible HDF5 episode storage

**Output:**
- HDF5 file: `data/gait_recordings/gait_session_YYYYMMDD_HHMMSS.h5`
- ~500-1000 timesteps per episode (60-second episodes)
- Fields: joint state, velocity commands, contact forces, ZMP, foot positions, etc.

---

## Installation and Running

### Prerequisites

- Isaac Sim 5.0 standalone installed
- Python packages: `h5py`, `fastapi`, `uvicorn`, `numpy`, `torch`

### Basic Execution

```bash
cd /workspaces/IsaacRobotics

# Run through Isaac Sim's Python interpreter
~/isaac-sim-standalone-5.0.0-linux-x86_64/python.sh \
  applications/spot_sensors_nav_data_collection.py \
  --num-episodes 100 \
  --episode-duration 60.0 \
  --dashboard-port 8001
```

### Key CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--num-episodes N` | 10 | Number of episodes to collect |
| `--episode-duration SEC` | 60.0 | Max duration per episode (seconds) |
| `--spawn-x X` | -8.0 | Robot initial spawn X coordinate |
| `--spawn-y Y` | 4.0 | Robot initial spawn Y coordinate |
| `--spawn-yaw YAW` | 0.0 | Robot initial spawn heading (radians) |
| `--arena-x-min/max` | -35.0 / 4.0 | Waypoint sampling X bounds |
| `--arena-y-min/max` | -25.0 / 28.0 | Waypoint sampling Y bounds |
| `--min-goal-dist M` | 2.0 | Minimum goal distance (meters) |
| `--max-goal-dist M` | 15.0 | Maximum goal distance (meters) |
| `--pos-thresh T` | 0.5 | Position reached threshold (meters) |
| `--yaw-thresh T` | 0.4 | Heading reached threshold (radians) |
| `--dashboard-port PORT` | 8001 | Web dashboard port |
| `--no-auto-start` | - | Launch without auto-starting (manual via dashboard) |

### Example: Auto-collect 200 Episodes in Warehouse

```bash
~/isaac-sim-standalone-5.0.0-linux-x86_64/python.sh \
  applications/spot_sensors_nav_data_collection.py \
  --num-episodes 200 \
  --episode-duration 60.0 \
  --arena-x-min -35 --arena-x-max 4 \
  --arena-y-min -25 --arena-y-max 28
```

### Access Web Dashboard

Open in browser:
```
http://localhost:8001
```

The dashboard displays collection status, episode count, robot trajectory, joint angles, and contact state in real-time.

---

## Architecture and Data Flow

### Dual Policy Stack

Gait data collection uses a hierarchical policy structure:

```
Goal position (x, y, yaw) — randomly sampled within bounds
    ↓
RandomWaypointGenerator
    ↓ (every 100 physics steps = 0.2s = 5 Hz decimation)
SpotNavigationPolicy (policy evaluation)
    obs: [lin_vel(3), gravity(3), pose_command(3 or 4)]
    → velocity command: [vx, vy, wz]
    ↓
SpotFlatTerrainPolicy (locomotion control)
    obs: [lin_vel(3), ang_vel(3), gravity(3), command(3), joint_pos(12),
          joint_vel(12), prev_action(12)] = 48-dim
    → joint position commands (12 DoF)
    ↓ (applied every physics step)
Isaac Sim physics engine (500 Hz, dt=0.002s)
    → robot body and environment interaction
    ↓
ContactSensorBridge (5 Hz contact collection)
    → foot contact forces, ZMP, foot positions
    ↓
GaitRecorder (buffer timestep to storage)
```

### Episode Lifecycle

```
start_collection() called (auto-start or dashboard)
    ↓
Initialize robot at fixed spawn position
    ↓
Sample random goal within arena bounds
    ↓
Navigation policy → locomotion policy → robot walks toward goal
    ↓
Goal reached? → sample new goal, continue walking
    ↓
Episode timeout (e.g., 60s)?
    ↓
Flush buffered gait data → HDF5 file
    ↓
Teleport robot back to fixed spawn position (no world.reset())
    ↓
Next episode starts
    ↓
All N episodes done? → Idle (sim stays running)
    ↓
Dashboard: Start Collection → repeats from spawn
```

### Collection Restart Behavior

- **First start**: sim already initialized, no world.reset()
- **Subsequent starts** (after stop): `world.reset(True)` called to reset sim to initial state
- **After each episode ends**: robot teleported to spawn via `set_world_pose()` + `set_joint_positions()` (no world.reset())
- **After all episodes complete**: sim stays running, robot idles, dashboard can trigger new collection

---

## Core Components

### 1. SpotGaitDataCollector (Main Class)

**Location**: `applications/spot_sensors_nav_data_collection.py`

Manages single robot (`/World/Spot`) and gait data collection.

**Key Methods:**

| Method | Description |
|--------|-------------|
| `__init__()` | Initialize robot USD prim, policies, navigation manager |
| `setup()` | Create ContactSensor prims, start dashboard server |
| `run()` | Main loop: sim steps → policy evaluation → data collection |
| `_on_physics_step()` | Called every physics step (500 Hz), query locomotion policy & collect state |
| `_flush_episode()` | Write buffered gait data to HDF5 |
| `_reset_to_spawn()` | Teleport robot to fixed spawn position and reinitialize |

**Key State Variables:**

```python
self._nav_decimation = 100        # Eval nav policy every 100 physics steps (5 Hz)
self._spawn_pos = np.array([...]) # Fixed spawn position (return each episode)
self._step_buf = []               # Current episode timestep buffer
self._current_wp = None           # Current goal waypoint (x, y, yaw)
self._base_cmd = np.zeros(3)      # Current velocity command [vx, vy, wz]
```

### 2. AutoCollectionManager (Thread-Safe State Machine)

**Location**: `applications/spot_sensors_nav_data_collection.py`

Manages episode lifecycle between Isaac Sim physics thread and FastAPI thread.

**Public API:**

```python
start_collection(config=None)  # Start collection, optionally update config
stop_collection()              # Stop collection
consume_world_reset_request()  # Check one-shot world.reset() flag
get_status()                   # Query collection status
get_config()                   # Get current config
update_config(config)          # Update config
is_collecting()                # Check if actively collecting
is_complete()                  # Check if all episodes done
```

**Returned Status:**
```python
{
    "active": bool,                 # Currently collecting?
    "current_episode": int,         # Current episode number
    "total_episodes": int,          # Total episodes to collect
    "episode_duration": float,      # Current episode elapsed time (seconds)
    "current_waypoint": [x, y, yaw],  # Current goal waypoint
    "goals_reached": int,           # Number of reached goals
}
```

### 3. RandomWaypointGenerator

**Location**: `applications/spot_sensors_nav_data_collection.py`

Samples random (x, y, yaw) goals within arena bounds.

**Features:**
- Min/max distance constraints (select goals not too close or far)
- Fallback to pure random sampling after 50 retries if no valid point found
- Heading (yaw) always randomly selected from [-π, π]

### 4. ContactSensorBridge (Contact Force Collection)

**Location**: `dashboard/backend/contact_sensor_bridge.py`

Collects foot contact forces using Isaac Sim 5.0's native ContactSensor API.

#### Critical: Lifecycle Order

```python
# Step 1: Create sensor prims (before world.reset())
bridge = ContactSensorBridge("/World/Spot")
bridge.pre_reset_setup()  # Apply PhysxContactReportAPI, create sensor prims

# Step 2: Activate sensor prims (call world.reset())
world.reset()  # ← MUST be after pre_reset_setup()

# Step 3: Initialize sensors (after world.reset() + spot.initialize())
bridge.post_reset_setup(physics_dt=0.002)

# Step 4: Update every physics step
bridge.update(dt)
forces = bridge.get_contact_forces()  # (4, 3) [Fx, Fy, Fz]
```

#### API

```python
# Get contact forces (Newtons)
get_contact_forces()  # → (4, 3) float32 FL/FR/HL/HR order

# Get air time (seconds)
get_air_times()       # → (4,) float32

# Check if foot in contact
get_foot_in_contact()  # → (4,) bool

# Compute ZMP (pressure-center method)
zmp = compute_zmp(foot_positions=(4,3), contact_forces=(4,3))
# → (2,) [x, y] or [nan, nan] (no contact)
```

#### Why ContactSensor?

The `PhysX.create_rigid_contact_view` API is broken in Isaac Sim 5.0:
- Always returns zeros
- Invalid `filter_patterns_expr` kwarg
- Doesn't work in standalone apps

**ContactSensor** is the official recommended API and works correctly in standalone apps.

#### Foot Order

| Index | Foot | USD Prim Name |
|-------|------|--------------|
| 0 | Front Left (FL) | `fl_foot` |
| 1 | Front Right (FR) | `fr_foot` |
| 2 | Hind Left (HL) | `hl_foot` |
| 3 | Hind Right (HR) | `hr_foot` |

### 5. GaitRecorder (HDF5 Storage)

**Location**: `dashboard/backend/gait_recorder.py`

Stores gait data to HDF5 in episode-based format.

**File Structure:**

```
data/gait_recordings/gait_session_20260319_154322.h5
├── attributes:
│   ├── created_at: "20260319_154322"
│   ├── robot_type: "SpotFlat"
│   ├── obs_dim: 48
│   └── action_dim: 12
└── episode_0000/
    ├── timestamps (N,)
    ├── obs (N, 48)
    ├── actions (N, 12)
    ├── commands (N, 3)
    ├── body_pose (N, 7)
    ├── body_lin_vel (N, 3)
    ├── body_ang_vel (N, 3)
    ├── com_position (N, 3)
    ├── zmp_position (N, 2)
    ├── foot_positions (N, 4, 3)
    ├── foot_contact_forces (N, 4, 3)
    ├── foot_air_time (N, 4)
    ├── head_position (N, 3)
    ├── joint_pos (N, 12)
    ├── joint_vel (N, 12)
    └── attributes:
        ├── episode_num: 0
        ├── start_time: 1710861822.5
        ├── end_time: 1710861882.5
        ├── duration_sec: 60.0
        ├── n_steps: 300
        ├── waypoints: "[[-15.2, 8.3, 0.5], ...]"
        └── arena_bounds: "{...}"
```

**API:**

```python
recorder = GaitRecorder(save_dir="data/gait_recordings")

# Start new session
session_file = recorder.start_session()  # Creates HDF5 file

# Start episode
episode_idx = recorder.start_episode(arena_bounds={...})

# Add one timestep per call
recorder.add_step({
    "timestamps": time.time(),
    "obs": np.zeros(48),
    "actions": np.zeros(12),
    # ... other fields
})

# Record visited waypoint
recorder.record_waypoint((x, y, yaw))

# End episode (flush buffer to HDF5)
metadata = recorder.stop_episode()

# Multiple episodes in one session
episode_idx = recorder.start_episode(...)
# ... add steps
recorder.stop_episode()

# Close session
recorder.close()
```

### 6. Dashboard Backend (FastAPI Server)

**Location**: `dashboard/backend/gait_main.py`

Provides REST API and WebSocket endpoints for web-based monitoring and control.

**REST Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/status` | Connection status, recording state, collection progress |
| POST | `/api/collection/start` | Start collection (body: `{num_episodes, episode_duration}`) |
| POST | `/api/collection/stop` | Stop collection |
| GET | `/api/collection/config` | Get current collection config |
| PUT | `/api/collection/config` | Update collection config |
| GET | `/api/episodes` | List recorded episodes |

**WebSocket:**

```
ws://localhost:8001/ws
```

Pushes at 10 Hz:
- Current gait state (joint angles, contact forces, foot positions, etc.)
- Collection status (current episode, goal, etc.)
- Sensor data

### 7. Dashboard Frontend (React)

**Location**: `dashboard/frontend/src/tabs/GaitDataCollection.tsx`

Web-based UI for real-time monitoring and collection control.

**Features:**
- Collection status display (offline/online, active/inactive)
- Episode progress bar
- Config inputs: Num Episodes, Episode Duration
- 2D trajectory chart (CoM path + ZMP trail)
- Joint angle chart (Hip Ab/Ad, Hip Flexion, Knee)
- Foot contact state indicators (FL/FR/HL/HR buttons)
- Foot contact force bar chart
- Episode history list

---

## Initialization Sequence

Correct initialization order is **critical**. ContactSensor especially must be created before `world.reset()` to activate properly.

```
main() starts
    ↓
SpotGaitDataCollector(...) created
    - Create robot USD prim (/World/Spot)
    - Load navigation policy
    - Load locomotion policy
    ↓
simulation_app.update()
    ↓
collector._world.reset()  # First reset
    - Load warehouse USD
    - Load robot USD
    ↓
simulation_app.update()
    ↓
collector.setup()
    - bridge.pre_reset_setup() called
        └─ Apply PhysxContactReportAPI to rigid bodies
        └─ Traverse USD stage to discover foot prims (fl_foot, fr_foot, hl_foot, hr_foot)
        └─ Create ContactSensor prims
    - Register physics callback (_on_physics_step)
    - Start FastAPI dashboard server
    ↓
collector._world.reset()  # Second reset
    - Activate ContactSensor prims (now reporting data)
    ↓
simulation_app.update()
    ↓
collector.run()  # Main loop starts
    - while not exit:
        ├─ world.step(render=True)
        ├─ _on_physics_step() callback (500 Hz)
        ├─ Eval nav policy (every 100 steps)
        ├─ Collect gait data
        ├─ On episode end: flush to HDF5
        └─ Teleport robot to spawn position
```

**Why Two world.reset() Calls?**

1. **First reset**: Loads USD stage (warehouse, Spot).
2. **setup() called**: ContactSensor prims are added to USD stage.
3. **Second reset**: Now-existing ContactSensor prims are activated and start reporting contact data.

Without the first reset, setup() cannot find foot prims because the stage isn't loaded yet. Without the second reset, ContactSensor never activates and always returns 0.

---

## Data Schema

### Observation Vector (obs) - 48-dim

Input to locomotion policy:

```
[lin_vel(3), ang_vel(3), gravity(3), command(3), joint_pos(12), joint_vel(12), prev_action(12)]
  0-2        3-5        6-8        9-11       12-23        24-35          36-47
```

| Field | Dim | Description |
|-------|-----|-------------|
| `lin_vel` | 3 | Linear velocity (world frame) |
| `ang_vel` | 3 | Angular velocity (world frame) |
| `gravity` | 3 | Gravity (robot body frame, for tilt sensing) |
| `command` | 3 | Velocity command [vx, vy, wz] (nav policy output) |
| `joint_pos` | 12 | Joint positions (4 legs × 3 DoF) |
| `joint_vel` | 12 | Joint velocities |
| `prev_action` | 12 | Previous timestep action |

### Action Vector (actions) - 12-dim

Output of locomotion policy (joint position deltas):

```
[FL_hip_aa, FL_hip_fe, FL_knee,
 FR_hip_aa, FR_hip_fe, FR_knee,
 HL_hip_aa, HL_hip_fe, HL_knee,
 HR_hip_aa, HR_hip_fe, HR_knee]
```

Each value is delta from default pose (0.2 scale).

### HDF5 Dataset Structure

Datasets within each episode group:

| Dataset | Shape | Type | Description |
|---------|-------|------|-------------|
| `timestamps` | (N,) | float64 | Unix timestamps |
| `obs` | (N, 48) | float32 | Policy observation vector |
| `actions` | (N, 12) | float32 | Joint position commands |
| `commands` | (N, 3) | float32 | Velocity command [vx, vy, wz] |
| `body_pose` | (N, 7) | float32 | Position(3) + quaternion(4) [w,x,y,z] |
| `body_lin_vel` | (N, 3) | float32 | Linear velocity (world frame) |
| `body_ang_vel` | (N, 3) | float32 | Angular velocity (world frame) |
| `com_position` | (N, 3) | float32 | Center of mass position (≈ robot body) |
| `zmp_position` | (N, 2) | float32 | Zero Moment Point [x, y] |
| `foot_positions` | (N, 4, 3) | float32 | Foot positions (world) FL/FR/HL/HR |
| `foot_contact_forces` | (N, 4, 3) | float32 | Ground reaction forces [Fx, Fy, Fz] |
| `foot_air_time` | (N, 4) | float32 | Air time per foot (seconds) |
| `head_position` | (N, 3) | float32 | Head position (no separate head link) |
| `joint_pos` | (N, 12) | float32 | Joint positions |
| `joint_vel` | (N, 12) | float32 | Joint velocities |

### Episode Metadata (HDF5 Attributes)

```python
episode_group.attrs = {
    "episode_num": 0,                    # Episode number
    "start_time": 1710861822.5,          # Start Unix timestamp
    "end_time": 1710861882.5,            # End Unix timestamp
    "duration_sec": 60.0,                # Episode duration
    "n_steps": 300,                      # Number of timesteps
    "waypoints": "[[-15.2, 8.3, 0.5], ...]",  # Visited waypoints (JSON)
    "arena_bounds": "{...}",             # Collection arena (JSON)
}
```

---

## Critical Design Decisions and Gotchas

### 1. Two world.reset() Calls

**Problem**: ContactSensor prims won't activate unless they exist in USD stage before reset.

**Solution**:
```python
collector._world.reset()  # First: load USD
collector.setup()         # Create ContactSensor prims
collector._world.reset()  # Second: activate prims
```

**Result**: Contact force data returns non-zero values only after second reset.

### 2. ContactSensor vs PhysX Tensor API

**Why ContactSensor?**

```python
# ❌ Doesn't work in Isaac Sim 5.0
contact_view = create_rigid_contact_view(...)  # Always returns 0
```

```python
# ✅ Correct way (Isaac Sim 5.0)
bridge = ContactSensorBridge(...)
bridge.pre_reset_setup()   # Create ContactSensor prims
world.reset()              # Activate
bridge.update(dt)          # Update every step
forces = bridge.get_contact_forces()  # Returns non-zero values
```

### 3. No world.reset() Between Episodes

**Optimization**: Don't reinitialize entire sim after each episode.

```python
# ❌ Slow (unnecessary resource usage)
world.reset()
robot.initialize()

# ✅ Fast (reset only robot state)
robot.set_world_pose(spawn_pos, spawn_quat)
robot.set_joint_positions(default_pos)
robot.set_joint_velocities(zeros)
```

For 50 episodes × 60s = 3000s ≈ 50 min:
- With `world.reset()`: 5-10s delay × 50 = 250-500s overhead
- With teleport: negligible overhead

### 4. Auto-Detect Navigation Policy Obs Dimension

Navigation policy accepts 3-dim or 4-dim pose commands:

```python
pose_dim = nav_policy.obs_dim - 6  # 6 = lin_vel(3) + gravity(3)

if pose_dim == 4:
    # [dx, dy, dz, heading_error]
    pose_command = np.array([pos_cmd[0], pos_cmd[1], pos_cmd[2], heading])
else:
    # [dx, dy, heading_error]
    pose_command = np.array([pos_cmd[0], pos_cmd[1], heading])
```

### 5. Body-Frame Coordinate Transform

Position commands are transformed to **robot body frame** (IsaacLab convention):

```python
# Goal in world frame
target_world = [goal_x, goal_y, robot_z]
target_vec = target_world - robot_pos

# Inverse yaw rotation
R_yaw_inv = [[cos(yaw), sin(yaw), 0],
             [-sin(yaw), cos(yaw), 0],
             [0, 0, 1]]

# Transform to body frame
pos_command_body = R_yaw_inv @ target_vec
```

### 6. ZMP Computation

Zero Moment Point computed as center of contact pressure:

```python
def compute_zmp(foot_positions, contact_forces, threshold=1.0):
    """
    foot_positions: (4, 3) - FL/FR/HL/HR
    contact_forces: (4, 3) - [Fx, Fy, Fz]
    threshold: min vertical force (N) — below ignored
    """
    fz = contact_forces[:, 2]
    mask = fz > threshold
    if not mask.any():
        return [nan, nan]

    fz_v = fz[mask]
    total = fz_v.sum()
    zmp_x = (fz_v * foot_positions[mask, 0]).sum() / total
    zmp_y = (fz_v * foot_positions[mask, 1]).sum() / total
    return [zmp_x, zmp_y]
```

### 7. Waypoint Sampling Constraints

Goals must satisfy:

```python
min_distance <= distance(robot, goal) <= max_distance
arena_bounds["x_min"] <= goal_x <= arena_bounds["x_max"]
arena_bounds["y_min"] <= goal_y <= arena_bounds["y_max"]
```

After 50 retries, fallback to unconstrained random sampling.

---

## Dashboard Usage

### Access Web Interface

```
http://localhost:8001
```

### GaitDataCollection Tab

#### Status Indicators

- **Offline/Connected**: Sim connection status
- **Collecting badge**: Currently collecting indicator

#### Collection Control

1. Enter **Episode Count**
2. Enter **Episode Duration** (seconds)
3. Click **Start Collection**
4. Monitor progress
5. Click **Stop Collection** (optional)

#### Real-Time Visualization

1. **Trajectory Chart**: CoM path (blue) + ZMP trail (red)
2. **Joint Angle Chart**: Selected joint angles over time
3. **Foot Contact State**: FL/FR/HL/HR on/off indicators
4. **Contact Force Bar**: Vertical force (N) per foot

#### Episode History

Completed episodes list:
- Episode number
- Duration
- Goals reached
- File saved

---

## Troubleshooting

### Contact Forces Always Zero

**Cause**: ContactSensor not activated

**Solution**:
```python
# 1. Verify pre_reset_setup() called
bridge.pre_reset_setup()

# 2. Verify world.reset() called
world.reset()

# 3. Verify post_reset_setup() called
bridge.post_reset_setup()

# 4. Check logs for:
# "[ContactBridge] Found foot prims (4): [...]"
```

### Cannot Find Foot Prims

**Cause**: USD stage missing foot prims or wrong names

**Solution**:
```bash
# Check foot prim names in Isaac Sim node graph
# Look for: fl_foot, fr_foot, hl_foot, hr_foot

# Verify spot_sensors.usd
grep -i "foot" assets/spot_sensors.usd
```

### Robot Won't Move

**Cause**: Policies not loaded or bad observations

**Solution**:
```bash
# Verify policy files exist
ls -la policies/spot_flat/models/policy.pt
ls -la policies/spot_nav/models/policy.pt

# Check logs for:
# "[GaitCollector] Nav policy loaded. obs_dim=..."
```

### Dashboard Won't Connect

**Cause**: FastAPI server didn't start or port in use

**Solution**:
```bash
# Check port
lsof -i :8001

# Use different port
python.sh applications/spot_sensors_nav_data_collection.py --dashboard-port 8002

# Access at
http://localhost:8002
```

### Episode Data Not Saved

**Cause**: `data/gait_recordings/` permissions issue

**Solution**:
```bash
mkdir -p data/gait_recordings
chmod 755 data/gait_recordings

# Verify files created
ls -la data/gait_recordings/
```

### Memory Leak or Performance Degradation

**Cause**: Large buffers not cleaned up

**Solution**:
```python
# Verify episode buffer cleared after episode end
self._step_buf = []

# Verify HDF5 file closed
recorder.close()
```

---

## Additional Resources

### Data Analysis Example (Python)

```python
import h5py
import numpy as np

# Open HDF5 file
with h5py.File('data/gait_recordings/gait_session_20260319_154322.h5', 'r') as f:
    # List episodes
    episodes = [k for k in f.keys() if k.startswith('episode_')]

    for ep_name in episodes[:5]:  # First 5 episodes
        ep = f[ep_name]

        # Metadata
        print(f"\n{ep_name}:")
        print(f"  Steps: {len(ep['timestamps'])}")
        print(f"  Duration: {ep.attrs['duration_sec']:.1f}s")
        print(f"  Goals reached: {len(ep.attrs['waypoints'])}")

        # Joint angle ranges
        joint_pos = ep['joint_pos'][:]
        print(f"  Joint pos range: [{joint_pos.min():.3f}, {joint_pos.max():.3f}]")

        # Max contact force
        forces = ep['foot_contact_forces'][:]
        max_force = forces[:, :, 2].max()
        print(f"  Max contact force: {max_force:.1f} N")
```

### Data Validation

```python
def validate_episode(h5_file, episode_idx):
    """Validate episode data integrity"""
    with h5py.File(h5_file, 'r') as f:
        ep = f[f'episode_{episode_idx:04d}']
        n_steps = len(ep['timestamps'])

        issues = []

        # Check all datasets same length
        for key in ep.keys():
            if len(ep[key]) != n_steps:
                issues.append(f"{key}: shape {ep[key].shape} != {n_steps}")

        # Check for all-NaN fields
        for key in ['zmp_position', 'foot_contact_forces']:
            if np.isnan(ep[key][:]).all():
                issues.append(f"{key}: all NaN")

        # Check timestamp monotonicity
        ts = ep['timestamps'][:]
        if not np.all(np.diff(ts) > 0):
            issues.append("Timestamps not monotonic")

        if issues:
            print(f"⚠ Episode {episode_idx} issues:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print(f"✓ Episode {episode_idx} valid")

        return len(issues) == 0
```
