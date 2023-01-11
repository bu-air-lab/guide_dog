# Training Force Tolerant Guide dog Controllers #

Based on code from: https://github.com/leggedrobotics/legged_gym

#Training

Run the following:

```
python3 legged_gym/scripts/train.py --task=guide_dog --trial_name=<any name here> --headless
```

#Testing

```
python3 legged_gym/scripts/play.py --task=guide_dog
```

#Pybullet Validation

```
cd pybullet_val
python3 -m scripts.play_bullet.py
```