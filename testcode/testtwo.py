flag = 0
# flag = 1
if flag:
    import numpy as np
    import torch

    arr = np.array([[1, 1], [2, 5], [7, 8]])
    arr = np.array(arr).astype(np.float32)
    arr = torch.from_numpy(arr).float()
    print(type(arr), "*", arr.shape)
    # seg_labels = np.eye(9)[png.reshape([-1])]
    print(arr.reshape([-1]))

    arr_numpy = arr.numpy().astype(np.uint8)

    print(type(arr), "&&", type(arr_numpy))

    re_arr = np.eye(10)[arr_numpy.reshape([-1])]
    print(re_arr)

    re_arr_ = re_arr.reshape(3, 2, 10)

    print(type(arr), arr.shape[0], arr.shape[1])
    print(type(arr_numpy), arr_numpy.shape)
    print(type(re_arr), re_arr.shape)
    print(type(re_arr_), re_arr_.shape)

    print(type(np.unique(re_arr_)[0]))

    print("***")


# arrone = np.array([[[1, 2], [1, 2]], []])



flag = 0
# flag = 1
if flag:
    from PIL import Image
    import numpy as np

    _img = Image.open("/home/user/Downloads/gc/pspnet/pspnet-pytorch-master_better/datasets/before/1.jpg").convert('RGB')

    print(_img.size[0])
    print(_img.size[1])

    img_arr = np.array(_img)
    print(img_arr.shape)



# flag = 0
flag = 1
if flag:
    import numpy as np
    import torch

    arr = np.array([[1, 1], [2, 5], [7, 8]])
    # seg_labels = np.eye(9)[png.reshape([-1])]
    print(arr.reshape([-1]))


    re_arr = np.eye(10)[arr.reshape([-1])]

    print(re_arr)

    re_arr_ = re_arr.reshape(3, 2, 10)

    print(type(arr), arr.shape)
    # print(type(re_arr), re_arr.shape)
    print(type(re_arr_), re_arr_.shape)

    # print(type(np.unique(re_arr_)[0]))

    print("***")

    print(arr)
    print(re_arr_)

    re_arr_ = np.array(re_arr_).astype(np.float32)
    re_arr_ = torch.from_numpy(re_arr_).float()
    print(re_arr_)