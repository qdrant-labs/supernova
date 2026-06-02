# Troubleshooting
Below are some common issues and tricks that I use to debug issues and also fix them. These are not meant to be comprehensive, but just a starting point for debugging.

## Expired credentials
Sometimes I'll get an error that looks like this, despite having **just** refreshed my AWS SSO credentials:
```
sky.exceptions.CloudUserIdentityError: Failed to get AWS user.
  Reason: [RuntimeError] Credentials were refreshed, but the refreshed credentials are still expired..
```
Most often the fix is to restart the `skypilot` server, which seems to cache credentials in memory and doesn't always pick up the refreshed credentials. To restart the server, run:
```bash
sky api stop && sky api start
```
Rerun your workload after that and it should pick up the new credentials.

## Jobs stuck between "pending" and "starting"
Sometimes I'll see a jobs stuck bouncing between the `PENDING` and `STARTING` states. This is usually because something failed when setting up the node, its failing, and then SkyPilot is retrying and failing again. An example might be a failure to install a dependency. To debug this, I inspect the controller logs:
```bash
sky jobs logs --controller # TODO: check if this is the right command
```
This will show you the logs from the controller, which is responsible for launching and managing jobs. You can also inspect the logs of the individual nodes, which might show you more details about why the job is failing to start

## Lingering nodes/workers
Sometimes, after all jobs are marked `SUCCEEDED` or `FAILED`, I'll see that there are still active EC2 instances running in the AWS console. I'm not sure why this is, but it seems like sometimes the cleanup process fails and leaves behind some nodes. To fix this, I usually just tell `skypilot` to stop all nodes:
```bash
sky jobs cancel --all
```
Or, if running a pool:
```bash
sky jobs pool down --all
```
This should terminate all nodes and clean up any lingering resources. Always double-check the AWS console to make sure there are no active instances after doing this, just to be safe.