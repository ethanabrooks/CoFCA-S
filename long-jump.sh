runs new \
--path="icml/long-jump/extended-olsk/0" \
--command="python ppo/control_flow/main.py                                       --gridworld --olsk --control-flow-types \"Subtask\" \"If\" \"Else\" \"While\" --conv-hidden-size=\"64\" --critic-type=\"h1\" --entropy-coef=\"0.015\" --eval-interval=\"100\" --eval-steps=\"500\" --failure-buffer-size=\"500\" --flip-prob=\"0.5\" --gate-coef=\"0.01\" --gate-conv-kernel-size=\"2\" --gate-hidden-size=\"16\" --gate-stride=\"2\" --hidden-size=\"64\" --hidden2=\"32\" --kernel-size=\"2\" --learning-rate=\"0.0025\" --lower-embed-size=\"64\" --lower-level-config=\"checkpoint/lower.json\" --lower-level-load-path=\"checkpoint/lower.pt\" --max-eval-lines=\"50\" --max-failure-sample-prob=\"0.3\" --max-lines=\"10\" --max-loops=\"3\" --max-nesting-depth=\"1\" --max-while-loops=\"10\" --max-world-resamples=\"20\" --min-eval-lines=\"10\" --min-lines=\"1\" --no-op-coef=\"0\" --no-op-limit=\"30\" --num-batch=\"1\" --num-conv-layers=\"0\" --num-edges=\"2\" --num-encoding-layers=\"0\" --num-layers=\"0\" --num-processes=\"150\" --num-steps=\"25\" --ppo-epoch=\"2\" --reject-while-prob=\"0.5\" --seed=\"0\" --stride=\"2\" --task-embed-size=\"32\" --term-on \"mine\" \"sell\" --time-to-waste=\"0\" --world-size=\"6\"  " \
--description="fix olsk and no_pointer" \
--path="icml/long-jump/extended-olsk/1" \
--command="python ppo/control_flow/main.py                                       --gridworld --olsk --control-flow-types \"Subtask\" \"If\" \"Else\" \"While\" --conv-hidden-size=\"64\" --critic-type=\"h1\" --entropy-coef=\"0.015\" --eval-interval=\"100\" --eval-steps=\"500\" --failure-buffer-size=\"500\" --flip-prob=\"0.5\" --gate-coef=\"0.01\" --gate-conv-kernel-size=\"2\" --gate-hidden-size=\"16\" --gate-stride=\"2\" --hidden-size=\"64\" --hidden2=\"32\" --kernel-size=\"2\" --learning-rate=\"0.0025\" --lower-embed-size=\"64\" --lower-level-config=\"checkpoint/lower.json\" --lower-level-load-path=\"checkpoint/lower.pt\" --max-eval-lines=\"50\" --max-failure-sample-prob=\"0.3\" --max-lines=\"10\" --max-loops=\"3\" --max-nesting-depth=\"1\" --max-while-loops=\"10\" --max-world-resamples=\"20\" --min-eval-lines=\"10\" --min-lines=\"1\" --no-op-coef=\"0\" --no-op-limit=\"30\" --num-batch=\"1\" --num-conv-layers=\"0\" --num-edges=\"2\" --num-encoding-layers=\"0\" --num-layers=\"0\" --num-processes=\"150\" --num-steps=\"25\" --ppo-epoch=\"2\" --reject-while-prob=\"0.5\" --seed=\"1\" --stride=\"2\" --task-embed-size=\"32\" --term-on \"mine\" \"sell\" --time-to-waste=\"0\" --world-size=\"6\"  " \
--description="fix olsk and no_pointer" \
--path="icml/long-jump/extended-olsk/2" \
--command="python ppo/control_flow/main.py                                       --gridworld --olsk --control-flow-types \"Subtask\" \"If\" \"Else\" \"While\" --conv-hidden-size=\"64\" --critic-type=\"h1\" --entropy-coef=\"0.015\" --eval-interval=\"100\" --eval-steps=\"500\" --failure-buffer-size=\"500\" --flip-prob=\"0.5\" --gate-coef=\"0.01\" --gate-conv-kernel-size=\"2\" --gate-hidden-size=\"16\" --gate-stride=\"2\" --hidden-size=\"64\" --hidden2=\"32\" --kernel-size=\"2\" --learning-rate=\"0.0025\" --lower-embed-size=\"64\" --lower-level-config=\"checkpoint/lower.json\" --lower-level-load-path=\"checkpoint/lower.pt\" --max-eval-lines=\"50\" --max-failure-sample-prob=\"0.3\" --max-lines=\"10\" --max-loops=\"3\" --max-nesting-depth=\"1\" --max-while-loops=\"10\" --max-world-resamples=\"20\" --min-eval-lines=\"10\" --min-lines=\"1\" --no-op-coef=\"0\" --no-op-limit=\"30\" --num-batch=\"1\" --num-conv-layers=\"0\" --num-edges=\"2\" --num-encoding-layers=\"0\" --num-layers=\"0\" --num-processes=\"150\" --num-steps=\"25\" --ppo-epoch=\"2\" --reject-while-prob=\"0.5\" --seed=\"2\" --stride=\"2\" --task-embed-size=\"32\" --term-on \"mine\" \"sell\" --time-to-waste=\"0\" --world-size=\"6\"  " \
--description="fix olsk and no_pointer" \
--path="icml/long-jump/extended-olsk/3" \
--command="python ppo/control_flow/main.py                                       --gridworld --olsk --control-flow-types \"Subtask\" \"If\" \"Else\" \"While\" --conv-hidden-size=\"64\" --critic-type=\"h1\" --entropy-coef=\"0.015\" --eval-interval=\"100\" --eval-steps=\"500\" --failure-buffer-size=\"500\" --flip-prob=\"0.5\" --gate-coef=\"0.01\" --gate-conv-kernel-size=\"2\" --gate-hidden-size=\"16\" --gate-stride=\"2\" --hidden-size=\"64\" --hidden2=\"32\" --kernel-size=\"2\" --learning-rate=\"0.0025\" --lower-embed-size=\"64\" --lower-level-config=\"checkpoint/lower.json\" --lower-level-load-path=\"checkpoint/lower.pt\" --max-eval-lines=\"50\" --max-failure-sample-prob=\"0.3\" --max-lines=\"10\" --max-loops=\"3\" --max-nesting-depth=\"1\" --max-while-loops=\"10\" --max-world-resamples=\"20\" --min-eval-lines=\"10\" --min-lines=\"1\" --no-op-coef=\"0\" --no-op-limit=\"30\" --num-batch=\"1\" --num-conv-layers=\"0\" --num-edges=\"2\" --num-encoding-layers=\"0\" --num-layers=\"0\" --num-processes=\"150\" --num-steps=\"25\" --ppo-epoch=\"2\" --reject-while-prob=\"0.5\" --seed=\"3\" --stride=\"2\" --task-embed-size=\"32\" --term-on \"mine\" \"sell\" --time-to-waste=\"0\" --world-size=\"6\"  " \
--description="fix olsk and no_pointer" \
--path="icml/long-jump/extended-olsk/4" \
--command="python ppo/control_flow/main.py                                       --gridworld --olsk --control-flow-types \"Subtask\" \"If\" \"Else\" \"While\" --conv-hidden-size=\"64\" --critic-type=\"h1\" --entropy-coef=\"0.015\" --eval-interval=\"100\" --eval-steps=\"500\" --failure-buffer-size=\"500\" --flip-prob=\"0.5\" --gate-coef=\"0.01\" --gate-conv-kernel-size=\"2\" --gate-hidden-size=\"16\" --gate-stride=\"2\" --hidden-size=\"64\" --hidden2=\"32\" --kernel-size=\"2\" --learning-rate=\"0.0025\" --lower-embed-size=\"64\" --lower-level-config=\"checkpoint/lower.json\" --lower-level-load-path=\"checkpoint/lower.pt\" --max-eval-lines=\"50\" --max-failure-sample-prob=\"0.3\" --max-lines=\"10\" --max-loops=\"3\" --max-nesting-depth=\"1\" --max-while-loops=\"10\" --max-world-resamples=\"20\" --min-eval-lines=\"10\" --min-lines=\"1\" --no-op-coef=\"0\" --no-op-limit=\"30\" --num-batch=\"1\" --num-conv-layers=\"0\" --num-edges=\"2\" --num-encoding-layers=\"0\" --num-layers=\"0\" --num-processes=\"150\" --num-steps=\"25\" --ppo-epoch=\"2\" --reject-while-prob=\"0.5\" --seed=\"4\" --stride=\"2\" --task-embed-size=\"32\" --term-on \"mine\" \"sell\" --time-to-waste=\"0\" --world-size=\"6\"  " \
--description="fix olsk and no_pointer"

