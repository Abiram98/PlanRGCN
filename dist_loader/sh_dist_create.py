import os

def create_sh_cmds(com_folder, dist_dir, cmd_dir, jar_file):
    files = [f"{com_folder}/{x}" for x in os.listdir(com_folder)]
    out_files = [f"{dist_dir}/u_{x}" for x in os.listdir(com_folder)]
    cpus = 22
    cmd_count = 0
    fw2 = open(f"{cmd_dir}/runner.sh",'w')
    fw = open(f"{cmd_dir}/cmd{cmd_count}.sh",'w')
    fw2.write(f"bash {cmd_dir}/cmd{cmd_count}.sh\n")
    for i,(f,o) in enumerate(zip(files, out_files)):
        if i % cpus == 0:
            fw.write("wait\n")
            fw.write("Finished After $SECONDS\n")
            fw.flush()
            fw.close()
            cmd_count += 1
            fw = open(f"{cmd_dir}/cmd{cmd_count}.sh",'w')
            fw2.write(f"bash {cmd_dir}/cmd{cmd_count}.sh\n")
            fw2.flush()
        
        fw.write(f"java -jar {jar_file} ged-opt --input-queryfile={f} --output-queryfile={o} &\n")
        fw.flush()
        
    fw.write("wait\n")
    fw.write("Finished After $SECONDS\n")
    fw.flush()
    fw.close()
    fw2.write(f"bash {cmd_dir}/cmd{cmd_count}.sh\n")
    fw2.flush()
    fw2.close()

jarfile='/qpp/qpp_features/sparql-query2vec/target/sparql-query2vec-0.0.1.jar'
com_folder = '/data/dbpedia_dist2bak/combinations2'
cmd_dir = '/data/dbpedia_dist2bak/cmds'
dist_dir = '/data/dbpedia_dist2bak/distances'
create_sh_cmds(com_folder, dist_dir, cmd_dir, jarfile)
