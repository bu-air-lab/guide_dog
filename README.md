# Training Force Tolerant Guide dog Controllers

Based on code from: https://github.com/leggedrobotics/legged_gym and https://github.com/leggedrobotics/rsl_rl

## Install

In main Directory, run:

```
pip3 install -e .
```

Next, run:

```
cd guide_dog_ppo
pip3 install -e .
```

## Training

Run the following:

```
python3 legged_gym/scripts/train.py --task=guide_dog --trial_name=<any name here> --headless
```

## Testing

```
python3 legged_gym/scripts/play.py --task=guide_dog
```

## Pybullet Validation

```
cd pybullet_val
python3 -m scripts.play_bullet.py
```