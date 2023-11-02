## Proactive Load Balancing for Task-based Applications on Distributed Memory Machines

The project is structured as follows.

### Chameleon framework and examples
This refers to the original source of [Chameleon](https://github.com/chameleon-hpc), a task-based programming framework for parallel applications on both shared and distributed memory. This framework is developed with the concept of task-based programming models and hybrid MPI+OpenMP. Along with it, there are some examples showing how to program an application with Chameleon. These are in the folders `/chameleon_patch/` and `/chameleon_examples/`.

### Proactive load balancing tool
Refers to the plugin tools built upon the Chameleon framework. In detail, the implementation is in `/proactlb_tools`, where we show how the idea of proactive load balancing works, including
* online load prediction
* proactive task offloading

### Work stealing and reactive load balancing simulation
Refers to the simulator for work stealing and reactive load balancing on distributed memory machines, as the related works of the proactive load balancing approach. The implementation of the simulator is in `/simdlb`, where we show the main factors of our simulator are delay time in task migration and the overhead of balancing operations before a task is decided for migration.

<!-- If everything is fine with dependencies, we could compile Chameleon by running the script.
``` Bash
# the working-dir is /compile_cham
source compile_chameleon_on_xyz.sh
``` -->

### Use cases
Refers to the real application, [Sam(oa)$^2$](https://github.com/meistero/Samoa). `Sam(oa)$^2$` is a software suit supporting for the oceanic applications in HPC. The concept of `Sam(oa)$^2$` is based on space-filling curves and adaptive mesh refinement. The configuration of `Sam(oa)$^2$` with a real context simulation, tsunami, is shown in `/usecases`.

### Others
The others refer to some python scripts in `/util_scripts` for visualizing performance data, logs. Futhermore, the source of the thesis "Proactive Load Balancing on Distributed Memory Machines" is in `/thesis`.


--------------------------------------------------------------
Compiling Samoa linked with Chameleon using Test-Runner

--------------------------------------------------------------
1. Download test-runner
Test-runner is a python-src for automatically compiling and submitting samoa-simulation jobs on HPC clusters (i.e., CoolMUC or SuperMUC-NG). It could be downloaded here https://gitlab.lrz.de/samoa/test-runner, or my modified version https://gitlab.lrz.de/minhchung/samoa-test-runner.
* To run test-runner, just type the command:
``` Bash
python3 test_runner.py <path-to-json-config-file> <options>
```
* Associated with the json-config file, we need to adapt other related properties in the following sections.
* Basically, there are some options for compiling samoa, generating slurm-scripts to submit jobs, and running those jobs automatically.

2. Adapt .json file about Samoa-Info
The sample json-config files could be found in ./json-configs. For example, if we take a look at osc_aderdgoptsamoa_cham.json in sub-folder coolmuc/, it's necessary to set:
``` json
"app_dir"       : "/dss/dsshome1/lxc0D/ra56kop/samoa",
"cache_dir"     : "/dss/dsshome1/lxc0D/ra56kop/chameleon/results/osc_test_16nodes_n20_d22_1tpr",
"command_dir"   : "/dss/dsshome1/lxc0D/ra56kop/chameleon/results/osc_test_16nodes_n20_d22_1tpr",

"build" :{
        "template"      : "/path/to/.../config_cham_aderdg_coolmuc.py",
        ...
}
"run":{
        "chameleon_lib" : { "cartesian" : ["intel_mig"]},
        "template"      : "/path/to/.../coolmuc_cham.slurm_template",
}
```
The first template file points to the configurations for compiling Samoa, such as linking with Chameleon or not, which solver for Samoa, ... There are some examples of this template in ./config/. The second template points to the configuration for job-description files of SLURM to generate and submit jobs, some examples of this config could be found in ./template/.

3. Adapt .py-template-config file for compiling Samoa
As mentioned above, just a few information need to be clarified here. For examples:
``` python
chameleon=1     # means that we are linking with Chameleon
asagi=False     # this belongs to Samoa stuffs
chameleon_dir="/path/to/installed-chameleon/"   # install-path to chameleon
...
```

4. Adapt .slurm_template-config file for `submitting` jobs
This file is to specify the commands for generating slurm-scripts, then we can use to submit jobs onto the clusters. Some examples of these configs could be found in `/template`.

5. Compile Sample with test-runner
* Finally, compile Samoa by command:
``` Bash
python3 test_runner.py /path/to/configs.json-file -b
```
* Where, `-b` for building, `-s` for generating submit-job scripts, `-r` for submitting jobs (run), `-o` for checking submit-job scripts, ...

