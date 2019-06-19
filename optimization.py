# This script aim to run  the model 9 times with different values of learning rate and batch size
# to optimize the model performance

import subprocess
def pause():
    input("Press the <ENTER> key to continue...")

learning_rates = [0.005, 0.01, 0.05]
train_batch_size = [50, 100, 200]
all_combinations = [(x,y) for x in learning_rates for y in train_batch_size]
program_name = "python"
arguments = ["retrain.py", r"--bottleneck_dir=C:\Users\USER1\Desktop\urban_sound\bottlenecks_mel_aug",
              "--how_many_training_steps=30000", "--model_dir=inception",
              r"--image_dir=C:\Users\USER1\Desktop\urban_sound\spectrograms_mel",
              "--validation_batch_size=-1", "--eval_step_interval=1000", "--use_augmentation"]

flag = 0
for lr, sz in all_combinations:
    command = [program_name]
    command.extend(arguments)
    if lr == 0.01 and sz ==100:
        print('Skip already exist mode') # Those are the default parameters so I've already used them.
        continue
    pause() # To catch when the run is over
    flag +=1
    lr_command = "--learning_rate=" + str(lr)
    sz_command = "--train_batch_size=" + str(sz)
    print('################################################')
    print(' Start combination -', flag,' LR = ', lr, ' SZ = ', sz )
    print('################################################ \n')
    command.append(lr_command)
    command.append(sz_command)
    subprocess.run(command)