# folder = "/home/vuong/PycharmProjects/hand-predict/hand-sign/data_raw_ver2"
# size = (400,400)
# import os
# from PIL import Image
#
# # for split in ["train", "test", "validation"]:
# #     for filename in os.listdir(os.path.join(folder, split)):
# #         if filename.__contains__(".jpeg"):
# #             filename = os.path.join(folder, split) + "/" + filename
# #             print(filename)
# #             image = Image.open(filename)
# #             image = image.resize(size)
# #             image.save(filename)
#
# def repair_xml(filepath):
#     with open(filepath, "r") as f:
#         lines = f.readlines()
#         del lines[4:9]
#     f.close()
#
#     with open(filepath, "w") as f:
#         f.writelines(lines)
#     f.close()
# for split in ["train", "test", "validation"]:
#     for filename in os.listdir(os.path.join(folder, split)):
#         if filename.__contains__(".xml"):
#             filename = os.path.join(folder, split) + "/" + filename
#             print(filename)
#             repair_xml(filename)