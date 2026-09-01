# 학습 속도 벤치마크

실제 페이지를 헤드리스 크로뮴에서 띄우고, 일반 수학 블록으로 만든 softmax 회귀
학습 그래프를 돌려 **step/s**를 측정합니다.

```bash
node bench/train-bench.mjs
```

주요 옵션:

| 옵션 | 기본값 | 설명 |
| --- | --- | --- |
| `--steps N` | `10000` | `반복` 블록의 학습 step 수 |
| `--runs N` | `3` | 측정에 사용할 실행 횟수 |
| `--warmup N` | `1` | 통계에서 제외할 워밍업 실행 횟수 |
| `--hidden N` | `0` | `0`이면 한 층, `N > 0`이면 ReLU 은닉층 하나를 추가 |
| `--classes a,b,c` | `cat,fish,house` | 사용할 Quick Draw 종류 |
| `--lr X` | `0.05` | 학습률 |
| `--seed N` | `1` | 가중치 초기화 시드 |
| `--json` | | 결과를 JSON으로 출력 |
| `--headed` | | 브라우저 창을 띄워서 확인 |

최적화 전후를 비교할 때 쓴 명령:

```bash
node bench/train-bench.mjs --steps 20000 --runs 5 --warmup 2
```

## 출력 읽는 법

```
종류 cat, fish, house · step 20,000 · 은닉층 없음 · lr 0.05 · seed 1 · warmup 2회 제외
  run 1: 20,557 step/s · 0.0486 ms/step
  ...
  중앙값 22,207 step/s · 최고 25,478 step/s
  가중치 체크섬 65448dec01716d81
  학습 후 held-out 손실 0.512683 (학습 전 1.110391)
```

- **step/s** — 컨테이너/CPU 상태에 따라 ±7% 정도 흔들립니다. 실행 간 비교는
  **최고값**을 기준으로 보는 편이 안정적입니다.
- **가중치 체크섬** — 학습이 끝난 뒤 모든 런타임 변수의 비트를 그대로 해싱한 값입니다.
  **오버헤드만 줄이는 최적화라면 이 값이 절대 변하면 안 됩니다.** 값이 바뀌었다면
  계산 결과가 달라진 것이므로 그 변경은 되돌려야 합니다.
- **held-out 손실** — 학습 구간에 쓰이지 않은 샘플 300장의 평균 손실입니다.
  학습 전에는 약 `1.11`, 20,000 step 뒤에는 약 `0.513`입니다. 벤치마크 그래프가
  실제로 학습하고 있다는 확인용 값입니다.

### 조용히 실패하지 않습니다

체크섬은 학습이 실제로 돌아갔을 때만 의미가 있습니다. 그래프가 첫 iteration에서
예외를 던지면 벽시계 시간만 짧아져서 step/s가 오히려 좋아 보이고, 체크섬은 학습 전
가중치의 값으로 고정됩니다. 그래서 다음을 검사합니다.

- 모든 노드의 `lastError`를 확인해서 하나라도 있으면 즉시 실패
- 학습 후 체크섬이 초기 가중치 체크섬과 같으면 실패
- held-out 손실이 학습 전보다 줄지 않으면 경고와 함께 exit code 1

`--steps`를 크게 바꿔도 체크섬이 그대로라면 학습이 돌지 않은 것입니다.

## 버퍼 재사용 검사

```bash
node bench/arena-check.mjs
```

> 이 검사는 현재 실행되지 않습니다. 아래 '아직 복구하지 않은 검사'를 보세요.

커널이 결과 버퍼를 재사용해도 계산 결과가 달라지지 않는지 확인합니다. 같은 그래프를
**재사용을 끄고 한 번, 켜고 한 번** 실행해 런타임 변수 상태를 비트 단위로 비교합니다.

```
OK   aliasingAccumulator  풀링 끔 52370d421c73f0f3 · 풀링 켬 52370d421c73f0f3
OK   userBlockTraining    풀링 끔 349c784bbe2e5e5a · 풀링 켬 349c784bbe2e5e5a
OK   matrixAccumulator    풀링 끔 613f1c38cfeda1fe · 풀링 켬 613f1c38cfeda1fe
OK   nestedRepeat         풀링 끔 4054022e11e8d6c9 · 풀링 켬 4054022e11e8d6c9
```

버퍼 재사용 버그는 예외를 던지지 않고 **조용히 틀린 값으로 학습**하기 때문에, 커널이나
버퍼 아레나를 건드린 뒤에는 이 검사를 반드시 돌리세요. 검사 그래프는 각각 다른 함정을 노립니다.

| 그래프 | 노리는 것 |
| --- | --- |
| `aliasingAccumulator` | `펼치기`·`값 보기`·`둘 다 계산`이 입력 버퍼를 그대로 넘기는 경로 |
| `userBlockTraining` | 사용자 블록을 통한 미분, `zerosLike`의 0 초기화 |
| `matrixAccumulator` | 회차를 넘어 살아남는 행렬이 `외적`에 들어가는 경우 |
| `nestedRepeat` | 중첩 반복의 회차 경계 |

## 공간 배열 연산 검사

```bash
node bench/spatial-math-check.mjs
```

> 이 검사는 현재 실행되지 않습니다. 아래 '아직 복구하지 않은 검사'를 보세요.

실제 브라우저 페이지에서 `배열 모양 바꾸기`와 `슬라이딩 창 펼치기`를 검사합니다.

검사 항목:

- reshape 모양 변경과 `-1` 자동 계산
- reshape의 zero-copy 동작
- 2×2 sliding-window unfold 순전파
- 겹치는 창에서 입력 gradient 누적
- zero padding
- `unfold → 행렬×벡터 → 합` 전체 그래프에서 커널과 입력 양쪽 자동미분

## 공간 연산 fusion 검사

```bash
node bench/spatial-fusion-check.mjs
```

> 이 검사는 현재 실행되지 않습니다. 아래 '아직 복구하지 않은 검사'를 보세요.

4필터 CNN에서 실제로 사용하는 패턴을 만들어 학습용 `unfold → 행렬×벡터` fusion을 검사합니다.

- 하나의 3×3 unfold를 네 합성곱 필터가 공유
- 각 필터 뒤의 2×2 / stride 2 평균 풀링
- fusion OFF/ON의 모든 변수 gradient를 Float32 비트 단위로 비교
- fusion된 matvec 개수 확인
- 같은 gradient pass의 평균 실행 시간을 함께 출력

fusion은 사용자에게 보이는 블록이나 수식을 바꾸지 않습니다. 학습 중에만 큰 patch 행렬과
patch-gradient 행렬의 실체화를 생략합니다. 필요하면 개발자 콘솔에서
`SPATIAL_TRAINING_FUSION.setDisabled(true)`로 기존 경로와 직접 비교할 수 있습니다.

## 필요한 것

- Playwright (전역 설치도 인식합니다. 다른 위치에 있으면 `PLAYWRIGHT_MODULE`로 지정)
- 크로뮴 (`PLAYWRIGHT_BROWSERS_PATH`가 설정돼 있으면 그대로 사용)
- `data/*.bin` — 저장소를 localhost로 서빙하므로 네트워크 접속은 필요 없습니다

## 측정 대상

`bench/train-bench.mjs`가 페이지 안에서 만드는 그래프입니다. 팔레트에서 손으로
조립하는 것과 같은 블록만 사용하고, 자동미분은 쓰지 않습니다. 모든 기울기는 이
파일이 직접 써 둔 역전파 정의이고 실행 순서도 `둘 다 계산` 연결로 직접 정합니다.

순전파 (`--hidden 0` 기본값):

```
데이터 샘플 → 그림값 → 펼치기 ─┐
                              ├→ 행렬×벡터 → 더하기(b) → z
변수 W(행렬) ─────────────────┘

z, 종류 벡터 → Softmax CE 사용자 블록 → loss
```

`--hidden N`을 주면 `행렬×벡터 → 더하기 → max(0,·)` 층이 앞에 하나 더 붙습니다.

역전파 (전부 손으로 쓴 정의):

| 블록 | 저장된 역전파 식 |
| --- | --- |
| Softmax CE 사용자 블록 | `dz = softmax(z) - y` |
| `더하기` | `da = g·1`, `db = g·1` |
| `행렬 × 벡터` | `dA = g ⊗ x`, `dx = Aᵀg` |
| `최댓값` | `db = g·[y = b]` (순전파 출력 `y`를 읽는 경로) |

```
loss → SoftmaxCE 역전파 → gz
gz  → 더하기 역전파 → gp, gb
gp  → 행렬×벡터 역전파 → gW  (첫 층의 dx는 목적지를 비워 두어 실행되지 않음)

gW → 곱하기(lr) → 빼기(W) → 값 바꾸기(W) ─┐
gb → 곱하기(lr) → 빼기(b) → 값 바꾸기(b) ─┴→ 둘 다 계산 → 반복
```

이렇게 하면 한 step 안에서 수동 역전파의 주요 경로가 모두 지나갑니다.

- 사용자 블록 순전파 중간값을 역전파 식이 재사용하는 경로
- 순전파 출력 `y`를 읽는 역전파 정의
- 저장하지 않는 gradient 가지의 제거
- gradient 변수의 즉시 쓰기와 다음 역전파 블록의 즉시 읽기

## 아직 복구하지 않은 검사

`arena-check.mjs`, `spatial-math-check.mjs`, `spatial-fusion-check.mjs`는 아직
제거된 자동미분(`미분` 블록, `primitiveVJP`, 학습용 fusion 런타임)에 의존하고
있어서 지금은 실행되지 않습니다. 실행하면 조용히 넘어가지 않고 오류로 멈춥니다.
수동 역전파 기준으로 다시 쓰는 작업이 남아 있습니다.

`weight-io-check.mjs`와 `bench/manual-backprop*-browser-smoke.html`은 정상
동작합니다.
