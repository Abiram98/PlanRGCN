from pathlib import Path
import load_balance.fifo_balancer as fifo
import load_balance.query_balancer as qbl
import configparser
import sys
import os

if __name__ == "__main__":
    #timeout -s 2 7200 python3 -m load_balance.main_balancer /path/to/config
    config = configparser.ConfigParser()
    config.read(sys.argv[1])
    sample_name=config['DATASET']['sample_name']
    scale=config['DATASET']['scale']
    url = config['DATABASE']['url']
    save_dir = config['DATASET']['save_dir']
    cls_field = config['DATASET']['true_field_name']
    std_file = config['DATASET']['stdout']
    add_lsq_url = config['DATASET'].getboolean('add_lsq_url')
    os.makedirs(Path(save_dir), exist_ok=True)
    with open(os.path.join(save_dir, std_file), 'w') as sys.stdout:
        match config['TASK']['taskName']:
            case "fifo":
                workers = int(config['LOADBALANCER']['FIFOWorkers'])
                fifo.main_balance_runner(sample_name,scale, url, save_dir=save_dir, cls_field=cls_field, n_workers=workers, add_lsq_url=add_lsq_url)
            case "qpp":
                fast_workers = int(config['LOADBALANCER']['FastWorkers'])
                med_workers = int(config['LOADBALANCER']['MediumWorkers'])
                slow_workers = int(config['LOADBALANCER']['SlowWorkers'])
                qbl.main_balance_runner(sample_name,scale, url, save_dir=save_dir, cls_field=cls_field, work_dict={
                    'fast': fast_workers,
                    'med' : med_workers,
                    'slow': slow_workers
                },add_lsq_url=add_lsq_url)

