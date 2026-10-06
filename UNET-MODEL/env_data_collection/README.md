# env_data_collection

Autonomous AirSim RGB / segmentation-mask collection for training the U-Net
perception module (`../train_unet.py`).

## Pipeline

1. **Edit `config_collection.py`**
   - `OBSTACLE_NAME_PATTERNS`: regexes matching your Unreal object names
     (check the World Outliner). Everything that doesn't match stays free
     space (segmentation ID 0).
   - `X_MIN/X_MAX/Y_MIN/Y_MAX`: bounding box covering the whole
     institute/campus map you want scanned.
   - `ALTITUDES`, `GRID_SPACING`, `YAW_SAMPLES_DEG`: coverage density.
   - `TARGET_PAIRS`: how many image/mask pairs to collect (5,000-10,000 is a
     good start).

2. **Calibrate the obstacle color** (one time, or whenever the environment's
   segmentation palette changes):

   ```
   env\Scripts\python.exe env_data_collection\calibrate_mask_color.py --x 0 --y 0 --z -10 --yaw_deg 90
   ```

   Point `--x/--y/--z/--yaw_deg` at a pose where an obstacle (building/tree)
   is actually in the camera's view. This writes
   `env_data_collection/obstacle_color.json` and a
   `calibration_preview.png` you should open to confirm the pose was
   sensible. If the script picked the wrong color from the printed list,
   rerun with `--pick_index <i>`.

3. **Collect data** (flies the whole bounding box autonomously):

   ```
   env\Scripts\python.exe env_data_collection\collect_data.py
   ```

   This applies the segmentation IDs, takes off, and flies a lawnmower
   sweep of the bounding box at every altitude in `ALTITUDES`, sampling
   several headings at each stop. Frames are only saved after the drone has
   moved `MIN_SAVE_DIST_M` or turned `MIN_SAVE_YAW_DEG` since the last saved
   frame, so slow motion doesn't flood the dataset with duplicates. Every
   `N`-th sweep column is written to `../data_val/` instead of `../data/` so
   validation comes from a spatially distinct area (see
   `VAL_COLUMN_MODULO`/`VAL_COLUMN_REMAINDER`), matching the paper's
   block-based train/val split guidance. If the sweep finishes before
   `TARGET_PAIRS` is reached, a randomized top-up pass keeps sampling extra
   poses in the bounding box.

   The script resumes cleanly: rerunning it continues numbering frames after
   whatever already exists in `../data/images` and `../data_val/images`.

4. **Check labels before training**:

   ```
   env\Scripts\python.exe env_data_collection\inspect_labels.py
   env\Scripts\python.exe env_data_collection\inspect_labels.py --data_dir ..\data_val
   ```

   Saves a contact sheet (`label_check.png`) of RGB | mask | red-overlay
   tiles and prints how many sampled masks are fully black/white, so you can
   catch a bad obstacle-color calibration or misaligned capture early.

5. **Train the U-Net**:

   ```
   env\Scripts\python.exe train_unet.py --data_dir data --epochs 20 --batch_size 16 --lr 0.001
   ```

   Optionally also point a run at `data_val` to monitor held-out loss.

6. **Retrain PPO/TRPO** against the newly trained U-Net, since the policy
   should learn from real occupancy grids rather than an untrained
   U-Net's noise:

   ```
   env\Scripts\python.exe train_ppo.py --timesteps 1000000 --alpha 0.01
   ```

## Files

- `config_collection.py` - all tunables (obstacle patterns, bounding box,
  altitudes, coverage/dedup thresholds, train/val split).
- `set_segmentation_ids.py` - applies `OBSTACLE_NAME_PATTERNS` -> ID 1 /
  everything else -> ID 0 via `simSetSegmentationObjectID`.
- `calibrate_mask_color.py` - one-time step to determine the segmentation
  camera's obstacle color and save it to `obstacle_color.json`.
- `mask_utils.py` - shared AirSim image capture + segmentation-color ->
  binary-mask thresholding used by both scripts above.
- `collect_data.py` - the autonomous lawnmower-sweep collector.
- `inspect_labels.py` - QA contact sheet + fully-black/white mask stats.

## Notes

- The collector treats a collision as a signal to climb to
  `COLLISION_RECOVERY_ALT_M` and skip that waypoint, rather than stopping
  the whole run, so a single bad waypoint near a building doesn't kill an
  overnight collection run.
- If your environment supports it, you can extend `collect_data.py` to call
  `client.simSetTimeOfDay(...)` / `simEnableWeather` + `simSetWeatherParameter`
  between passes for lighting/weather variety (Step 2's last bullet); left
  out by default since not every AirSim settings.json enables weather.
