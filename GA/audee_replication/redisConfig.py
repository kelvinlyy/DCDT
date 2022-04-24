import os
import time
import subprocess

redis_dir_path = '/data/yylaiai/redis/redis-stable/src'

def run():
    try:
        # start redis server
        subprocess.Popen([os.path.join(redis_dir_path, 'redis-server')])
        
        time.sleep(10)

        # increase the usable size of redis database memory
        cli_path = os.path.join(redis_dir_path,'redis-cli')
        os.system(f'{cli_path} config set proto-max-bulk-len 50gb')
        os.system(f'{cli_path} config set client-query-buffer-limit 50gb')
        print("Configured")
    except Exception as e:
        print(e)