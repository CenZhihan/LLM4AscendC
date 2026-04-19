# 编译构建

## 环境准备<a name="prepare&install"></a>
本项目支持源码编译，在源码编译前，需要确保已经安装驱动、固件和CANN软件（Ascend-cann-toolkit和Ascend-cann-ops）。

  软件安装方式请根据如下描述进行选择：

| 安装方式 | 说明 |使用场景|
| :--- | :--- | :--- |
| 使用WebIDE安装 | WebIDE可提供在线直接运行的昇腾环境。目前仅适用于Atlas A2系列产品，ARM架构。当前可提供单机算力，**默认安装最新商发版CANN软件包和固件/驱动包。**<br>**对于master分支的使用，需要参考场景三安装最新社区包。**| 适用于没有昇腾设备的开发者。|
| 使用Docker部署 | Docker镜像是一种CANN高效部署方式，目前仅适用于Atlas A2系列产品，OS仅支持Ubuntu操作系统。**默认安装最新商发版CANN软件包和固件/驱动包。**|适用有昇腾设备，需要快速搭建环境的开发者。|
| 手动安装软件包 | - |适用有昇腾设备，想体验手动安装CANN包或体验最新master分支能力的开发者。|


### 场景一：使用WebIDE安装

对于无环境的用户，可直接使用WebIDE开发平台，即“**一站式开发平台**”，该平台为您提供在线可直接运行的昇腾环境，环境中已安装必备的软件包，无需手动安装。
> **说明**：
<br>环境默认安装最新商发版CANN包和固件/驱动包，源码下载时注意与软件配套。更多关于开发平台的介绍请参考[LINK](https://gitcode.com/org/cann/discussions/54)。

1. 进入开源项目，单击“`云开发`”按钮，使用已认证过的华为云账号登录。若未注册或认证，请根据页面提示进行注册和认证。

   <img src="./figures/cloudIDE.png" alt="云平台"  width="750px" height="90px">

2. 根据页面提示创建并启动云开发环境，单击“`连接 > WebIDE `”进入算子一站式开发平台，开源项目的资源默认在`/mnt/workspace`目录下。

   <img src="./figures/webIDE.png" alt="云平台"  width="1000px" height="150px">


### 场景二：使用Docker部署

>**说明：**
<br>镜像文件比较大，正常网速下，下载时间约为5-10分钟，请您耐心等待。
<br>环境默认安装最新商发版CANN包和固件/驱动包，源码下载时注意与软件配套。

**1.安装固件和驱动**：请参考[CANN软件安装指南](https://www.hiascend.com/document/redirect/CannCommunityInstWizard)。

**2.下载CANN镜像**

- 步骤1：以root用户登录宿主机。确保宿主机已安装Docker引擎（版本1.11.2及以上）。

- 步骤2：从[昇腾镜像仓库](https://www.hiascend.com/developer/ascendhub/detail/17da20d1c2b6493cb38765adeba85884)拉取已预集成的CANN软件包及所需依赖的镜像。命令如下，根据实际架构选择：

    ```bash
    # 示例：拉取ARM架构的CANN开发镜像
    docker pull --platform=arm64 swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-beta.1-910b-ubuntu22.04-py3.11
    # 示例：拉取X86架构的CANN开发镜像
    docker pull --platform=amd64 swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-beta.1-910b-ubuntu22.04-py3.11
    ```

**3.运行Docker**
<br>拉取镜像后，需要以特定参数启动容器，以便容器内能访问宿主的昇腾设备。

```bash
docker run --name cann_container --device /dev/davinci0 --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc -v /usr/local/dcmi:/usr/local/dcmi -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info -v /etc/ascend_install.info:/etc/ascend_install.info -it swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-beta.1-910b-ubuntu22.04-py3.11 bash
```
| 参数 | 说明 | 注意事项 |
| :--- | :--- | :--- |
| `--name cann_container` | 为容器指定名称，便于管理。 | 可自定义。 |
| `--device /dev/davinci0` | 核心：将宿主机的NPU设备卡映射到容器内，可指定映射多张NPU设备卡。 | 必须根据实际情况调整：`davinci0`对应系统中的第0张NPU卡。请先在宿主机执行 `npu-smi info`命令，根据输出显示的设备号（如`NPU 0`, `NPU 1`）来修改此编号。|
| `--device /dev/davinci_manager` | 映射NPU设备管理接口。 | - |
| `--device /dev/devmm_svm` | 映射设备内存管理接口。 | - |
| `--device /dev/hisi_hdc` | 映射主机与设备间的通信接口。 | - |
| `-v /usr/local/dcmi:/usr/local/dcmi` | 挂载设备容器管理接口（DCMI）相关工具和库。 | -|
| `-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi` | 挂载`npu-smi`工具。 | 使容器内可以直接运行此命令来查询NPU状态和性能信息。|
| `-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/` | 关键挂载：将宿主机的NPU驱动库映射到容器内。 | -|
| `-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info` | 挂载驱动版本信息文件。 | -|
| `-v /etc/ascend_install.info:/etc/ascend_install.info` | 挂载CANN软件安装信息文件。 |- |
| `-v /home/your_dir:/home/your_dir` | 挂载宿主机的一个路径到容器中。 | 可自选使用。 |
| `-it` | `-i`（交互式）和 `-t`（分配伪终端）的组合参数。 |- |
| `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-beta.1-910b-ubuntu22.04-py3.11` | 指定要运行的Docker镜像。 |请确保此镜像名和标签（tag）与你通过`docker pull`拉取的镜像完全一致。 |
| `bash` | 容器启动后立即执行的命令。 |- |

### 场景三：手动安装软件包

**场景1：体验master版本能力或基于master版本进行开发**

如果您想体验**master分支最新能力**，单击[下载链接](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master)获取软件包，按照如下步骤进行安装。更多安装指导请参考[CANN软件安装指南](https://www.hiascend.com/document/redirect/CannCommunityInstWizard)。

1. 安装固件和驱动。

    运行算子时必须安装驱动与固件，若仅编译算子，可跳过本操作，安装指导请参考[CANN软件安装指南](https://www.hiascend.com/document/redirect/CannCommunityInstWizard)。
   
2. 安装社区版CANN toolkit包。

    ```bash
    # 确保安装包具有可执行权限
    chmod +x Ascend-cann-toolkit_${cann_version}_linux-${arch}.run
    # 安装命令
    ./Ascend-cann-toolkit_${cann_version}_linux-${arch}.run --install --install-path=${install_path}
    ```

3. 安装社区版CANN ops包。

    运行算子前必须安装本包，若仅编译算子，可跳过本操作。

    ```bash
    # 确保安装包具有可执行权限
    chmod +x Ascend-cann-${soc_name}-ops_${cann_version}_linux-${arch}.run
    # 安装命令
    ./Ascend-cann-${soc_name}-ops_${cann_version}_linux-${arch}.run --install --install-path=${install_path}
    ```
    - \$\{cann\_version\}：表示CANN包版本号。
    - \$\{arch\}：表示CPU架构，如aarch64、x86_64。
    - \$\{soc\_name\}：表示NPU型号名称。
    - \$\{install\_path\}：表示指定安装路径，需要与toolkit包安装在相同路径，root用户默认安装在`/usr/local/Ascend`目录，非root用户默认安装在`$HOME/Ascend`目录。

**场景2：体验已发布版本能力或基于已发布版本进行开发**

如果您想体验**官网正式发布的CANN包**能力，请访问[CANN官网下载中心](https://www.hiascend.com/cann/download)，选择对应版本CANN软件包（仅支持CANN 8.5.0及后续版本）进行安装。


## 环境验证

安装完CANN包后，需验证环境和驱动是否正常。

-   **检查NPU设备**：
    ```bash
    # 运行npu-smi，若能正常显示设备信息，则驱动正常
    npu-smi info
    ```
-   **检查CANN安装**：
    ```bash
    # 查看CANN Toolkit的version字段提供的版本信息（默认路径安装），<arch>表示CPU架构（aarch64或x86_64）。WebIDE场景下，请将/usr/local替换为/home/developer。
    cat /usr/local/Ascend/cann/<arch>-linux/ascend_toolkit_install.info
    # 查看CANN ops的version字段提供的版本信息（默认路径安装）。WebIDE场景下，请将/usr/local替换为/home/developer。
    cat /usr/local/Ascend/cann/<arch>-linux/ascend_ops_install.info
    ```

## 环境变量配置

按需选择合适的命令使环境变量生效。
```bash
# 默认路径安装，以root用户为例（非root用户，将/usr/local替换为${HOME}）
source /usr/local/Ascend/cann/set_env.sh
# 指定路径安装
# source ${install_path}/cann/set_env.sh
```

## 源码编译步骤

### 下载源码

开发者可通过如下命令下载本仓源码：

```bash
# 下载项目源码，以master分支为例
git clone https://gitcode.com/cann/asc-devkit.git
``` 

### 安装依赖

 以下所列仅为本开源仓源码编译用到的依赖，其中python、gcc、cmake的安装方法请参见配套版本的[用户手册](https://hiascend.com/document/redirect/CannCommunityInstDepend)，选择安装场景后，参见“安装CANN > 安装依赖”章节进行相关依赖的安装。

   - python >= 3.9.0

   - gcc >= 7.3.0

   - cmake >= 3.16.0

   - patch

     Ubuntu系统执行以下命令安装：
     ```
     sudo apt install patch
     ```
     EulerOS操作系统执行以下命令安装：
     ```
     sudo yum install patch
     ```

### 编译安装<a name="compile&install"></a>

1. 编译

   本开源仓提供一键式编译安装能力。

   方式一：进入本开源仓代码根目录，执行如下命令：

   ```bash
   bash build.sh --pkg
   ```

   方式二：用户也可使用离线下载功能，手动下载[makeself源码包](https://gitcode.com/cann-src-third-party/makeself/releases/download/release-2.5.0-patch1.0/makeself-release-2.5.0-patch1.tar.gz)存放至自定义目录PATH_TO_DOWNLOAD，进入本开源仓代码根目录，执行如下命令：

   ```bash
   bash build.sh --pkg --cann_3rd_lib_path={PATH_TO_DOWNLOAD} # PATH_TO_DOWNLOAD为自定义下载目录
   ```

   编译完成后会在`build_out`目录下生成cann-asc-devkit_*<cann_version>*_linux-*\<arch\>*.run软件包。
2. 安装

   在开源仓根目录下执行下列命令，将编译生成的run包安装到默认路径（/usr/local/Ascend）；或安装到指定的CANN包的装包路径（${install_path}），同时会覆盖原CANN包中的Ascend C内容。

   ```bash
   # 切换到run包生成路径下
   cd build_out
   # 默认路径安装run包
   ./cann-asc-devkit_<cann_version>_linux-<arch>.run --full
   # 指定路径安装run包
   ./cann-asc-devkit_<cann_version>_linux-<arch>.run --full --install-path=${install_path}
   ```


### UT验证

#### 安装依赖

- pytest >= 8.0.0

    执行以下命令安装：
    ```bash
    pip3 install pytest
    ```

- coverage >= 4.5.4

    执行以下命令安装：
    ```bash
    pip3 install coverage
    ```

- lcov >= 1.16(仅在执行覆盖率统计场景需要)

    下载[lcov源码](https://gitcode.com/cann-src-third-party/lcov/releases/download/v1.16/lcov-1.16.tar.gz)后，执行以下命令安装：
    ```bash
    tar -xf lcov-1.16.tar.gz
    cd lcov-1.16
    make install                         # root用户安装
    # sudo make install                  # 非root用户安装
    ```

#### 执行
方式一：在开源仓根目录执行下列命令，将按各模块依次批跑tests目录下的用例，得到结果日志，用于看护编译是否正常。

```bash
bash build.sh --adv_test                         # 批跑tests目录下adv_api里的用例
bash build.sh --basic_test_one                   # 批跑tests目录下basic_api part-one里的用例
bash build.sh --basic_test_two                   # 批跑tests目录下basic_api part-two里的用例
bash build.sh --basic_test_three                 # 批跑tests目录下basic_api part-three里的用例
```

方式二：用户也可使用离线下载功能，手动下载[三方库源码包](#开源第三方软件依赖)存放至自定义目录PATH_TO_DOWNLOAD，在开源仓根目录执行下列命令，同时批跑执行各模块的用例。

```bash
# 以PATH_TO_DOWNLOAD为自定义下载目录为例
bash build.sh --adv_test --cann_3rd_lib_path={PATH_TO_DOWNLOAD}          # 批跑tests目录下adv_api里的用例
bash build.sh --basic_test_one --cann_3rd_lib_path={PATH_TO_DOWNLOAD}    # 批跑tests目录下basic_api part-one里的用例
bash build.sh --basic_test_two --cann_3rd_lib_path={PATH_TO_DOWNLOAD}    # 批跑tests目录下basic_api part-two里的用例
bash build.sh --basic_test_three --cann_3rd_lib_path={PATH_TO_DOWNLOAD}  # 批跑tests目录下basic_api part-three里的用例
```

#### 开源第三方软件依赖

在执行ut时，依赖的第三方开源软件列表如下：

| 开源软件 | 版本 | 下载地址 |
|---|---|---|
| googletest | 1.14.0 | [googletest-1.14.0.tar.gz](https://gitcode.com/cann-src-third-party/googletest/releases/download/v1.14.0/googletest-1.14.0.tar.gz) |
| boost | 1.87.0 | [boost_1_87_0.tar.gz](https://gitcode.com/cann-src-third-party/boost/releases/download/v1.87.0/boost_1_87_0.tar.gz) |
| mockcpp | 2.7 | [makeself-release-2.5.0-patch1.tar.gz](https://gitcode.com/cann-src-third-party/mockcpp/releases/download/v2.7-h3/mockcpp-2.7.tar.gz) |
| mockcpp_patch | 2.7 | [mockcpp-2.7_py3-h3.patch](https://gitcode.com/cann-src-third-party/mockcpp/releases/download/v2.7-h3/mockcpp-2.7_py3-h3.patch) |

