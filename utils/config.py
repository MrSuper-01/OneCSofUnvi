# This file is used to configure the training parameters for each task
class Config_US30K:
    # This dataset contain all the collected ultrasound dataset
    data_path = "../../dataset/SAMUS/"  
    save_path = "./checkpoints/SAMUS/"
    result_path = "./result/SAMUS/"
    tensorboard_path = "./tensorboard/SAMUS/"
    load_path = save_path + "/xxx.pth"
    save_path_code = "_"

    workers = 1                         # number of data loading workers (default: 8)
    epochs = 200                        # number of total epochs to run (default: 400)
    batch_size = 8                     # batch size (default: 4)
    learning_rate = 5e-4                # iniial learning rate (default: 0.001)
    momentum = 0.9                      # momntum
    classes = 2                         # thenumber of classes (background + foreground)
    img_size = 256                      # theinput size of model
    train_split = "train"               # the file name of training set
    val_split = "val"                   # the file name of testing set
    test_split = "test"                 # the file name of testing set
    crop = None                         # the cropped image size
    eval_freq = 1                       # the frequency of evaluate the model
    save_freq = 2000                    # the frequency of saving the model
    device = "cuda"                     # training device, cpu or cuda
    cuda = "on"                         # switch on/off cuda option (default: off)
    gray = "yes"                        # the type of input image
    img_channel = 1                     # the channel of input image
    eval_mode = "mask_slice"                 # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "SAM"

# -------------------------------------------------------------------------------------------------
class Config_TN3K:
    data_path = "../../dataset/SAMUS/" 
    data_subpath = "../../dataset/SAMUS/ThyroidNodule-TN3K/" 
    save_path = "./checkpoints/TN3K/"
    result_path = "./result/TN3K/"
    tensorboard_path = "./tensorboard/TN3K/"
    load_path = save_path + "/xxx.pth"
    save_path_code = "_"

    workers = 1                  # number of data loading workers (default: 8)
    epochs = 400                 # number of total epochs to run (default: 400)
    batch_size = 8               # batch size (default: 4)
    learning_rate = 1e-4         # initial learning rate (default: 0.001)
    momentum = 0.9               # momentum
    classes = 2                  # the number of classes (background + foreground)
    img_size = 256               # the input size of model
    train_split = "train-ThyroidNodule-TN3K"  # the file name of training set
    val_split = "val-ThyroidNodule-TN3K"     # the file name of testing set
    test_split = "test-ThyroidNodule-TN3K"     # the file name of testing set
    crop = None                  # the cropped image size
    eval_freq = 1                # the frequency of evaluate the model
    save_freq = 2000               # the frequency of saving the model
    device = "cuda"              # training device, cpu or cuda
    cuda = "on"                  # switch on/off cuda option (default: off)
    gray = "yes"                 # the type of input image
    img_channel = 1              # the channel of input image
    eval_mode = "mask_slice"        # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "SAM"

class Config_BUSI:
    # This dataset is for breast cancer segmentation
    data_path = "../../dataset/SAMUS/"
    data_subpath = "../../dataset/SAMUS/Breast-BUSI/"   
    save_path = "./checkpoints/BUSI/"
    result_path = "./result/BUSI/"
    tensorboard_path = "./tensorboard/BUSI/"
    load_path = save_path + "/xxx.pth"
    save_path_code = "_"

    workers = 1                         # number of data loading workers (default: 8)
    epochs = 400                        # number of total epochs to run (default: 400)
    batch_size = 8                     # batch size (default: 4)
    learning_rate = 1e-4                # iniial learning rate (default: 0.001)
    momentum = 0.9                      # momntum
    classes = 2                         # thenumber of classes (background + foreground)
    img_size = 256                      # theinput size of model
    train_split = "train-Breast-BUSI"   # the file name of training set
    val_split = "val-Breast-BUSI"       # the file name of testing set
    test_split = "test-Breast-BUSI"     # the file name of testing set
    crop = None                         # the cropped image size
    eval_freq = 1                       # the frequency of evaluate the model
    save_freq = 2000                    # the frequency of saving the model
    device = "cuda"                     # training device, cpu or cuda
    cuda = "on"                         # switch on/off cuda option (default: off)
    gray = "yes"                        # the type of input image
    img_channel = 1                     # the channel of input image
    eval_mode = "mask_slice"                 # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "SAM"

class Config_CAMUS:
    # This dataset is for breast cancer segmentation
    data_path = "./dataset/SAMUS/CAMUS"  # 
    data_subpath = "CAMUS" 
    save_path = "./checkpoints/CAMUS/"
    result_path = "./result/CAMUS/"
    tensorboard_path = "./tensorboard/CAMUS/"
    load_path = save_path + "xxx.pth"
    save_path_code = "_"

    workers = 1                         # number of data loading workers (default: 8)
    epochs = 400                        # number of total epochs to run (default: 400)
    batch_size = 8                     # batch size (default: 4)
    learning_rate = 1e-4                # iniial learning rate (default: 0.001)
    momentum = 0.9                      # momntum
    classes = 4                        # thenumber of classes (background + foreground)
    img_size = 256                      # theinput size of model
    train_split = "train-EchocardiographyLA-CAMUS"   # the file name of training set
    val_split = "val-EchocardiographyLA-CAMUS"       # the file name of testing set
    test_split = "test-Echocardiography-CAMUS"     # the file name of testing set # HMCQU
    crop = None                         # the cropped image size
    eval_freq = 1                       # the frequency of evaluate the model
    save_freq = 2000                    # the frequency of saving the model
    device = "cuda"                     # training device, cpu or cuda
    cuda = "on"                         # switch on/off cuda option (default: off)
    gray = "yes"                        # the type of input image
    img_channel = 1                     # the channel of input image
    eval_mode = "camusmulti"                 # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "SAM"

class Config_EchoNet():
    data_path = "/EchoNet"  #
    data_subpath = "EchoNet" 
    save_path = "./checkpoints/EchoNet/"
    result_path = "./result/EchoNet/"
    tensorboard_path = "./tensorboard/EchoNet/"
    load_path = save_path + "SAMUS_10181927_95_0.9257182998911371.pth"
    save_path_code = "_"

    workers = 8                         # number of data loading workers (default: 8)
    epochs = 100                        # number of total epochs to run (default: 400)
    batch_size = 16                     # batch size (default: 4)
    learning_rate = 1e-4                # iniial learning rate (default: 0.001)
    momentum = 0.9                      # momntum
    classes = 2                        # thenumber of classes (background + foreground)
    img_size = 256                      # theinput size of model
    train_split = "echonet_train_filenames"   # the file name of training set
    val_split = "echonet_val_filenames"       # the file name of testing set
    test_split = "echonet_test_filenames"     # the file name of testing set # HMCQU
    crop = None                         # the cropped image size
    eval_freq = 1                       # the frequency of evaluate the model
    save_freq = 2000                    # the frequency of saving the model
    device = "cuda"                     # training device, cpu or cuda
    cuda = "on"                         # switch on/off cuda option (default: off)
    gray = "yes"                        # the type of input image
    img_channel = 1                     # the channel of input image
    eval_mode = "echonet"                 # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "SAM"

class Config_EchoNet_Video():
    data_path = "./EchoNet/echocycle"  #
    data_subpath = "EchoNet" 
    save_path = "./checkpoints/EchoNet/"
    result_path = "./result/EchoNet/"
    tensorboard_path = "./tensorboard/EchoNet/"
    load_path = save_path + "SAMUS_10081703_24_0.9262574595178807.pth"
    save_path_code = "_"

    workers = 8                         # number of data loading workers (default: 8)
    epochs = 30                        # number of total epochs to run (default: 400)
    batch_size = 16                     # batch size (default: 4)
    learning_rate = 1e-4                # iniial learning rate (default: 0.001)
    momentum = 0.9                      # momntum
    classes = 2                        # thenumber of classes (background + foreground)
    img_size = 256                      # theinput size of model
    train_split = "train"   # the file name of training set
    val_split = "val"       # the file name of testing set
    test_split = "test"     # the file name of testing set # HMCQU
    crop = None                         # the cropped image size
    eval_freq = 1                       # the frequency of evaluate the model
    save_freq = 2000                    # the frequency of saving the model
    device = "cuda"                     # training device, cpu or cuda
    cuda = "on"                         # switch on/off cuda option (default: off)
    gray = "yes"                        # the type of input image
    img_channel = 1                     # the channel of input image
    eval_mode = "echonet"                 # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "SAM"

class Config_CAMUS_Video():
    data_path = "./dataset/SAMUS/CAMUS"  # 
    data_subpath = "CAMUS" 
    save_path = "./checkpoints/CAMUS/"
    result_path = "./result/CAMUS/"
    tensorboard_path = "./tensorboard/CAMUS/"
    load_path = save_path + "SAMUS_10081703_24_0.9262574595178807.pth"
    save_path_code = "_"

    workers = 8                         # number of data loading workers (default: 8)
    epochs = 100                        # number of total epochs to run (default: 400)
    batch_size = 16                     # batch size (default: 4)
    learning_rate = 1e-4                # iniial learning rate (default: 0.001)
    momentum = 0.9                      # momntum
    classes = 2                        # thenumber of classes (background + foreground)
    img_size = 256                      # theinput size of model
    train_split = "train"   # the file name of training set
    val_split = "val"       # the file name of testing set
    test_split = "test"     # the file name of testing set # HMCQU
    crop = None                         # the cropped image size
    eval_freq = 1                       # the frequency of evaluate the model
    save_freq = 2000                    # the frequency of saving the model
    device = "cuda"                     # training device, cpu or cuda
    cuda = "on"                         # switch on/off cuda option (default: off)
    gray = "yes"                        # the type of input image
    img_channel = 1                     # the channel of input image
    eval_mode = "camus"                 # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "SAM"

class Config_CAMUS_Video_Full():
    data_path = "./dataset/SAMUS/CAMUS_full"  # 
    data_subpath = "CAMUS_full" 
    save_path = "./checkpoints/CAMUS_full/"
    result_path = "./result/CAMUS_full/"
    tensorboard_path = "./tensorboard/CAMUS_full/"
    load_path = save_path + "SAMUS_10081703_24_0.9262574595178807.pth"
    save_path_code = "_"

    workers = 8                         # number of data loading workers (default: 8)
    epochs = 100                        # number of total epochs to run (default: 400)
    batch_size = 16                     # batch size (default: 4)
    learning_rate = 1e-4                # iniial learning rate (default: 0.001)
    momentum = 0.9                      # momntum
    classes = 2                        # thenumber of classes (background + foreground)
    img_size = 256                      # theinput size of model
    train_split = "train"   # the file name of training set
    val_split = "val"       # the file name of testing set
    test_split = "test"     # the file name of testing set # HMCQU
    crop = None                         # the cropped image size
    eval_freq = 1                       # the frequency of evaluate the model
    save_freq = 2000                    # the frequency of saving the model
    device = "cuda"                     # training device, cpu or cuda
    cuda = "on"                         # switch on/off cuda option (default: off)
    gray = "yes"                        # the type of input image
    img_channel = 1                     # the channel of input image
    eval_mode = "camus"                 # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "SAM"

class Config_Breast():  # YAOHENG修改1121
    data_path = "/dahuafs/userdata/99309212/OtherProject/2025220/out_breast"#"/root/classify/out_breast"  # 数据集路径，指向乳腺数据集  # 修改路径
    data_subpath = "/dahuafs/userdata/99309212/OtherProject/2025220/out_breast" #"/root/project/classify/out_breast"  # 修改路径
    save_path = "./checkpoints/Breast/"
    result_path = "./result/Breast/"
    tensorboard_path = "./tensorboard/Breast/"
    load_path = save_path + "MemSAM_110721_15_0.926.pth"  # 预训练模型的路径，可自行修改
    save_path_code = "_"

    workers = 8                          # 数据加载的线程数
    epochs = 100                          # 总的训练轮数
    batch_size = 4                      # 每次训练的数据量
    learning_rate = 1e-4                 # 初始学习率
    momentum = 0.9                       # 动量
    classes = 2                          # 类别数（前景+背景）
    img_size = 256                       # 输入图像的尺寸
    train_split = "train"                # 训练集的文件名
    val_split = "val"                    # 验证集的文件名
    test_split = "test"                  # 测试集的文件名
    crop = None                          # 裁剪后的图像大小
    eval_freq = 1                        # 模型评估的频率
    save_freq = 2000                     # 保存模型的频率
    device = "cuda"                      # 训练设备，cpu 或 cuda
    cuda = "on"                          # 是否使用 CUDA
    gray = "yes"                         # 输入图像类型
    img_channel = 1                      # 输入图像的通道数
    eval_mode = "slice"                  # 评估模型的模式，切片级别或者病人级别
    pre_trained = False                  # 是否加载预训练模型
    mode = "train"                       # 模型训练或测试模式
    visual = False                        # 是否可视化
    modelname = "MemSAM"                 # 使用的模型名称




# ==================================================================================================

def get_config(task="US30K"):
    if task == "US30K":
        return Config_US30K()
    elif task == "TN3K":
        return Config_TN3K()
    elif task == "BUSI":
        return Config_BUSI()
    elif task == "CAMUS":
        return Config_CAMUS()
    elif task == "EchoNet":
        return Config_EchoNet()
    elif task == "EchoNet_Video":
        return Config_EchoNet_Video()
    elif task == "CAMUS_Video_Full":
        return Config_CAMUS_Video_Full()
    elif task == "Breast":  # YAOHENG修改1121
        return Config_Breast()
    else:
        assert("We do not have the related dataset, please choose another task.")