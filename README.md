# PyBullet Evaluation of Force Tolerant Controller

## Init
Copy all the code of this repo into /guide_dog/pybullet_val/

## To run
- To change start timestep, duration, or force, go to lines 215-226 of /bullet_env/bullet_env.py
- The main scripts are /scripts/motion_imitation/mpc_controller/play_mpc.py and /scripts/play_bullet.py
- The force comparing script is check_force_data.py
- Another important file is /scripts/motion_imitation/mpc_controller/play_mpc.py but we don't need to tinker with it anymore

To run MPC controller:
In /bullet_env/bullet_env.py, comment out line 53 and uncomment line 54, in other words, self.time_step should be equal to 0.001
Then, run
```/guide_dog/pybullet_val$ python -m scripts.motion_imitation.mpc_controller.play_mpc```

To run Policy:
In /bullet_env/bullet_env.py, comment out line 54 and uncomment line 53, in other words, self.time_step should be equal to 0.005
Then, run
```/guide_dog/pybullet_val$ python -m scripts.play_bullet```

To compare force data:
Run
```/guide_dog/pybullet_val$ python check_force_data.py```