## This is main script of UrbanSound classification project

# First I  I downloaded tensorflow script using:
# curl -O https://raw.githubusercontent.com/tensorflow/tensorflow/r1.1/tensorflow/examples/image_retraining \
  #           /retrain.py

# Then, I edited retrain.py and run it several times with different set of arguments.


import wav_to_png
import subprocess

# Create spectrograms
wav_to_png.my_main()

#  retrain.py was executed through command line with those arguments:

# # Step 1. mel spectrograms

command = ["python"]
args = ["retrain.py", r"--bottleneck_dir=C:\Users\USER1\Desktop\urban_sound\bottlenecks_mel",
               "--how_many_training_steps=30000", "--model_dir=inception",
               r"--image_dir=C:\Users\USER1\Desktop\urban_sound\spectrograms_mel",
              "--validation_batch_size=-1", "--eval_step_interval=1000"]
command.extend(args)
subprocess.run(command)

# # Step 2. raw spectrograms

command = ["python"]
args = ["retrain.py", r"--bottleneck_dir=C:\Users\USER1\Desktop\urban_sound\bottlenecks_raw",
               "--how_many_training_steps=30000", "--model_dir=inception",
               r"--image_dir=C:\Users\USER1\Desktop\urban_sound\spectrograms_raw",
              "--validation_batch_size=-1", "--eval_step_interval=1000"]
command.extend(args)
subprocess.run(command)

# # Step 3. 100,000 training steps

command = ["python"]
args = ["retrain.py", r"--bottleneck_dir=C:\Users\USER1\Desktop\urban_sound\bottlenecks_mel",
               "--how_many_training_steps=100000", "--model_dir=inception",
               r"--image_dir=C:\Users\USER1\Desktop\urban_sound\spectrograms_mel",
              "--validation_batch_size=-1", "--eval_step_interval=1000"]
command.extend(args)
subprocess.run(command)

# # Step 4. Data augmentation
command = ["python"]
args = ["retrain.py", r"--bottleneck_dir=C:\Users\USER1\Desktop\urban_sound\bottlenecks_mel_aug",
               "--how_many_training_steps=30000", "--model_dir=inception",
               r"--image_dir=C:\Users\USER1\Desktop\urban_sound\spectrograms_mel",
              "--validation_batch_size=-1", "--eval_step_interval=1000", "--create_aug_files"]
command.extend(args)
subprocess.run(command)

# # Step 5. Optimization
import optimization

# # Step 6. Test
command = ["python"]
args = ["retrain.py", r"--bottleneck_dir=C:\Users\USER1\Desktop\urban_sound\bottlenecks_mel_aug",
           "--how_many_training_steps=50000", "--model_dir=inception",
           r"--image_dir=C:\Users\USER1\Desktop\urban_sound\spectrograms_mel",
          "--validation_batch_size=-1", "--eval_step_interval=1000", "--use_augmentation",
          "--learning_rate=0.05", "--train_batch_size=200", "--do_test"]
command.extend(args)
subprocess.run(command)

