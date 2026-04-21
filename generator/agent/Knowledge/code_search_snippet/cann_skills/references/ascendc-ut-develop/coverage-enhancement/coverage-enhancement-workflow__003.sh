cd {PROJECT_DIR} && bash build.sh -u --ophost --ops='{operator_name}' --soc='{soc_version}' --cov 2>&1 | tee {outputdir}/{op_name}/log/round_0.log
