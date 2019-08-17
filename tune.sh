nice exp \
  --cuda-deterministic \
  --log-dir=/home/ethanbro/tune_results
  --run-id=tune/maiden \
  --num-processes="300" \

  --tune \
  --redis-address 141.212.113.250:6379 \

  # env variables
  --subtask="AnswerDoor" \
  --subtask="AvoidDog" \
  --subtask="ComfortBaby" \
  --subtask="KillFlies" \
  --subtask="MakeFire" \
  --subtask="WatchBaby" \
  --test "WatchBaby"  "KillFlies" "MakeFire"\
  --test "AnswerDoor"  "KillFlies" "AvoidDog"\
  --test "AnswerDoor"  "MakeFire" "AvoidDog"\
  --n-active-subtasks="3" \
  --time-limit="30" \

  # intervals
  --eval-interval="100" \
  --log-interval="10" \
  --save-interval="300" \

  # dummies
  --env="" \
  --num-batch="-1" \
  --num-steps="-1" \
  --seed="-1" \
  --entropy-coef="-1" \
  --hidden-size="-1" \
  --num-layers="-1" \
  --learning-rate="-1" \
  --ppo-epoch="-1"
