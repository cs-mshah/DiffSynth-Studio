# SLURM

To launch a job use the `wrapper.sh`, by running it from the root directory of the repository. Available commands:

```shell
# ./slurm/wrapper.sh <script_name> <log_dir_for_slurm (optional)>

# test run to view gpu and environment info
./slurm/wrapper.sh submit_job_test



To view logs while job is running, use:
```shell
# To show the last 10 lines of <file> and to wait for <file> to grow:
tail -f <log_file>
```

During the run, log files will be at the root directory. Once the run is over, they are automatically moved to the `LOG_DIR` passed as input. Default `LOG_DIR` is `slurm/logs`.