What is this project about
=========
This is just a project of llama2 demos. 


How to build
============
1. download llama2 model
```shell
    # 到FB申请模型使用权，会通过邮件得到临时下载地址，24小时有效
    # https://ai.meta.com/resources/models-and-libraries/llama-downloads/
    # 可以用脚本下载
    llama/download.sh
```

2. install python 3.10+
```shell
    # update pip
    python -m pip install --upgrade pip
```

3. 安装匹配的cuda及pytorch版本
```shell
    python -m venv venv
    . venv/bin/activate

    # 查询cuda版本及pytorch版本
    # https://pytorch.org/get-started/locally/
    # https://developer.nvidia.com/cuda-toolkit-archive

    # 下载所需cuda，比如118
    # 安装对应版本的pytorch
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

    # 测试一下
    import torch
    print(torch.cuda.is_available())
```

4. install packages
```shell
    # 必须：安装requirements
    python -m venv venv
    . venv/bin/activate
    pip install -r requirements.txt
```

5. run
```shell
    python -m venv venv
    python chat_demo01.py
```

Reference
=========
