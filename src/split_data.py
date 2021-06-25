# from collections import defaultdict
# import random

# def split_train(file_path, ratio):
#     f = open(file_path)
#     imgs = defaultdict(list)
#     for i in f.readlines():
#         label = i.strip().split('/')[0]
#         filename = i.strip()
#         imgs[label].append(filename)
#         # img_list.append(i.strip())
#     # print(imgs.values(), len(imgs))
#     print(file_path)
#     for c in imgs:
#         print(c, len(imgs[c]))
#     print('TOTAL', sum([len(i) for i in imgs.values()]))


# # split_train('./data/lists/train.txt', 0.2)
# # print()
# # split_train('./data/lists/test.txt', 0.2)

# def count(split_file_path, class_file_path):
#     train = []
#     test = []
#     f = open(split_file_path)
#     for i in f.readlines():
#         l = i.split()
#         if l[1] == '1':
#             train.append(l[0])
#         else:
#             test.append(l[0])
#     print(f"TRAIN {len(train)} TEST {len(test)}")
#     f.close()

#     f = open(class_file_path)
#     train_class, test_class = defaultdict(list), defaultdict(list)
#     all_class = defaultdict(list)
#     for i in f.readlines():
#         img_id, class_id = i.split()
#         all_class[class_id].append(img_id)
#         if img_id in train:
#             train_class[class_id].append(img_id)
#         else:
#             test_class[class_id].append(img_id)
#     f.close()

#     train_class = defaultdict(list)
#     val_class = defaultdict(list)
#     test_class = defaultdict(list)
#     for c in all_class:
#         values = all_class[c]
#         #42:12:6
#         random.shuffle(values)
#         train_num = int(len(values) * 0.7)
#         val_num = int(len(values) * 0.2)
#         test_num = len(values) - train_num - val_num

#         train_class[c] = values[:train_num]
#         val_class[c] = values[train_num: train_num+val_num]
#         test_class[c] = values[:test_num]


#     # val_class = defaultdict(list)
#     # for c in test_class:
#     #     values = test_class[c]
#     #     val_num = int(len(values) * 0.2)
#     #     # val_class[c] = random.sample(values, val_num)

#     #     random.shuffle(values)
#     #     val_class[c] = values[:val_num]
#     #     test_class[c] = values[val_num:]
#     for c in val_class.keys():
#         print(f'TRAIN: {len(train_class[c])} VAL: {len(val_class[c])} TEST: {len(test_class[c])}')
    
#     f = open("data/CUB_200_2011/train_val_test_split.txt", 'w')
#     for c in train_class:
#         for i in train_class[c]:
#             f.write(f"{i} 0\n")
#     for c in val_class:
#         for i in val_class[c]:
#             f.write(f"{i} 1\n")
#     for c in test_class:
#         for i in test_class[c]:
#             f.write(f"{i} 2\n")
#     f.close()

#     # f = open("./CUB_200_2011/CUB_200_2011/train.txt", 'w')
#     # for c in train_class:
#     #     for i in train_class[c]:
#     #         f.write(i)
#     #         f.write("\n")
#     # f.close()

#     # f = open("./CUB_200_2011/CUB_200_2011/val.txt", 'w')
#     # for c in val_class:
#     #     for i in val_class[c]:
#     #         f.write(i)
#     #         f.write("\n")
#     # f.close()

#     # f = open("./CUB_200_2011/CUB_200_2011/test.txt", 'w')
#     # for c in test_class:
#     #     for i in test_class[c]:
#     #         f.write(i)
#     #         f.write("\n")
#     # f.close()

#     # print("TRAIN")
#     # print([ f"{class_id}: {len(train_class[class_id])}" for class_id in train_class])
#     # print("TEST")
#     # print([ f"{class_id}: {len(test_class[class_id])}" for class_id in test_class])


# # count("data/CUB_200_2011/train_test_split.txt", "data/CUB_200_2011/image_class_labels.txt")

# def process_attribute(file_path):
#     f = open(file_path)
#     attribute = dict()
#     for i in f.readlines():
#         i = i.split()
#         id, attribute_id, label, certainty = i[:4]
#         id = int(id)
#         certainty = int(certainty)
#         if label == '0':
#             label = -1
#         else:
#             label = 1
#         if certainty == 1:
#             label *= 0
#         elif certainty == 2:
#             label *= 0
#         # elif certainty == 3:
#         #     label *= 0.9

#         if id in attribute:
#             attribute[id][int(attribute_id)-1] = str(label)
#         else:
#             attribute[id] = [0] * 312
#             attribute[id][int(attribute_id)-1] = str(label)
#     f.close()
#     f = open("data/CUB_200_2011/attributes/attributes_adjusted.txt", 'w')
#     for id in sorted(attribute.keys()):
#         f.write(','.join(attribute[id]))
#         f.write('\n')
#     f.close()

# # process_attribute("data/CUB_200_2011/attributes/image_attribute_labels.txt")

# def process_attribute_original(file_path):
#     f = open(file_path)
#     attribute = dict()
#     for i in f.readlines():
#         i = i.split()
#         id, attribute_id, label, certainty = i[:4]
#         id = int(id)
#         certainty = int(certainty)
#         if label == '0':
#             label = -1
#         else:
#             label = 1
#         if certainty == 1:
#             label *= 0
#         elif certainty == 2:
#             label *= 0
#         elif certainty == 3:
#             label *= 0.9

#         if id in attribute:
#             attribute[id][int(attribute_id)-1] = str(label)
#         else:
#             attribute[id] = [0] * 312
#             attribute[id][int(attribute_id)-1] = str(label)
#     f.close()
#     f = open("data/CUB_200_2011/attributes/attributes_original.txt", 'w')
#     for id in sorted(attribute.keys()):
#         f.write(','.join(attribute[id]))
#         f.write('\n')
#     f.close()
# # process_attribute_original("data/CUB_200_2011/attributes/image_attribute_labels.txt")

# import os, cv2
# def find_min_shape(dir):
#     directories = os.listdir(dir)
#     h = 0
#     w = 0
#     count = 0
#     onedim = 0
#     for folder in directories:
#         d = os.path.join(dir, folder)
#         if os.path.isdir(d):
#             for img in os.listdir(d):
#                 i = cv2.imread(os.path.join(dir, folder,img))
#                 h += i.shape[0]
#                 w += i.shape[1]
#                 c = i.shape[2]
#                 if c==1:
#                     onedim += 1
#                 count += 1
#                 # print(i.shape)
#     print(onedim)
#     print(h/count, w/count)
# # find_min_shape("./data/CUB_200_2011/CUB_200_2011/images")


# # import dataloader
# # from torch.utils.data import DataLoader
# # from torchvision import transforms
# # mean=[0.485, 0.456, 0.406]
# # std=[0.229, 0.224, 0.225]
# # transform = transforms.Compose([
# #     transforms.ToTensor(),
# #     transforms.Normalize(mean=mean, std=std),
# #     transforms.Resize((386, 468)),
# # ])
# # train = dataloader.CubImageDataset('../data', 0, transform=transform)


# # loader = DataLoader(train, batch_size=64)
# # for x, y in loader:
# #     print(y)
# #     break

# import matplotlib.pyplot as plt
# def plot(train_loss_list, train_acc_list, valid_loss_list, valid_acc_list, save_dir):
#     fig1, ax1 = plt.subplots(1)
#     fig2, ax2 = plt.subplots(1)
    
#     ax1.set_title("Loss")
#     ax1.plot(train_loss_list, label='Train Loss', color='green')
#     ax1.plot(valid_loss_list, label='Validation Loss', color='red')
#     ax1.legend()
#     fig1.savefig(save_dir+'loss.png')
    
#     ax2.set_title("Accuracy")
#     ax2.plot(train_acc_list, label='Train Accuracy', color='green')
#     ax2.plot(valid_acc_list, label='Validation Accuracy', color='red')
#     ax2.legend()
#     fig2.savefig(save_dir+'acc.png')
#     print(save_dir)

# # plot([1,2,3,10], [0.1, 0.2, 0.5, 0.6], [1,3,4,6], [0.1, 0.1, 0.3, 0.4], './')

# def count_train_val_test(file_path):
#     f = open("data/CUB_200_2011/train_val_test_split.txt")
#     train=0
#     val=0
#     test=0
#     for l in f.readlines():
#         _, c = l.split()
#         if c =='0':
#             train+= 1
#         elif c=='1':
#             val += 1
#         else:
#             test += 1
#     print(train, val, test)

# count_train_val_test("data/CUB_200_2011/train_val_test_split.txt")

def process_attribute(attribute_path, class_path):
    f = open(attribute_path)
    attribute = dict()
    for i in f.readlines():
        i = i.split()
        id, attribute_id, label, certainty = i[:4]
        id = int(id)
        certainty = int(certainty)

        if id in attribute:
            attribute[id][int(attribute_id)-1] = int(label)
        else:
            attribute[id] = [0] * 312
            attribute[id][int(attribute_id)-1] = int(label)
    f.close()

    f = open(class_path)
    img_to_class = dict()
    for i in f.readlines():
        i = i.split()
        img_id, class_id = i
        class_id = int(class_id)
        img_id = int(img_id)
        if class_id not in img_to_class:
            img_to_class[class_id] = [img_id]
        else:
            img_to_class[class_id].append(img_id)
    f.close()

    for c in img_to_class:
        img_ids = img_to_class[c]
        img_count = len(img_ids)
        print(f'current class {c} with {img_count} images')
        attributes_count = [0]*312
        for id in img_ids:
            for attr in range(312):
                attributes_count[attr] += attribute[id][attr]
        for attr in range(312):
            if attributes_count[attr] >= img_count / 2:
                for id in img_ids:
                    attribute[id][attr] = 1
            else:
                for id in img_ids:
                    attribute[id][attr] = 0
    write_attribute(attribute)

def write_attribute(attribute):
    f = open("/Users/zhouqirui/Documents/Stanford/2021 Spring/CS231n/Project/data/CUB_200_2011/attributes/attributes_majority_votes.txt", 'w')
    for id in sorted(attribute.keys()):
        f.write(','.join([str(i) for i in attribute[id]]))
        # f.write(','.join(attribute[id]))
        f.write('\n')
    f.close()
    print("Success")

# process_attribute('/Users/zhouqirui/Documents/Stanford/2021 Spring/CS231n/Project/data/CUB_200_2011/attributes/image_attribute_labels.txt', '/Users/zhouqirui/Documents/Stanford/2021 Spring/CS231n/Project/data/CUB_200_2011/image_class_labels.txt')

def check():
    f = open("/Users/zhouqirui/Documents/Stanford/2021 Spring/CS231n/Project/data/CUB_200_2011/attributes/attributes_majority_votes.txt")
    img_to_attr = []
    for i in f.readlines():
        i = i.split(',')
        img_to_attr.append(i)
    f.close()

    f = open('/Users/zhouqirui/Documents/Stanford/2021 Spring/CS231n/Project/data/CUB_200_2011/image_class_labels.txt')
    img_to_class = dict()
    for i in f.readlines():
        i = i.split()
        img_id, class_id = i
        class_id = int(class_id)
        img_id = int(img_id)-1
        if class_id not in img_to_class:
            img_to_class[class_id] = [img_id]
        else:
            img_to_class[class_id].append(img_id)
    f.close()

    class_to_attr = dict()
    for c in img_to_class:
        img_index = img_to_class[c]
        ref = img_to_attr[img_index[0]]
        class_to_attr[c] = ref
        # print(ref==img_to_attr[img_index[1]])
        # if (all([img_to_attr[index] == ref for index in img_index])):
        #     print(f'class {c} pass')
        # else:
        #     print(f'class {c} failed!')
        #     # return
    attr_to_count = dict()
    for attr in range(312):
        count = 0
        for c in class_to_attr:
            # print(class_to_attr[c])
            if class_to_attr[c][attr] == '1':
                count += 1
        # return
        attr_to_count[attr] = count
    print(attr_to_count) #attr start from 0
    print([v for v in attr_to_count.values() if v >= 10])
    print(len([v for v in attr_to_count.values() if v >= 10]))
    attributes_to_keep = []
    for attr in attr_to_count:
        if attr_to_count[attr] >= 10:
            attributes_to_keep.append(attr)
    print(attributes_to_keep)
    print(len(attributes_to_keep))

    print(len(class_to_attr[1]))
    for c in class_to_attr:
        newAtt=[i for index, i in enumerate(class_to_attr[c]) if index in attributes_to_keep]
        class_to_attr[c] = newAtt
    print(len(class_to_attr[1]))


    for c in img_to_class:
        indexes = img_to_class[c]
        for i in indexes:
            img_to_attr[i] = class_to_attr[c]
    print(len(img_to_attr[0]), len(img_to_attr))
    #img_to_attr
    # for index, c in enumerate(img_to_attr):
        #img_id starts from 0
        




    f = open("/Users/zhouqirui/Documents/Stanford/2021 Spring/CS231n/Project/data/CUB_200_2011/attributes/attributes_majority_votes_with_selection.txt", 'w')
    for l in img_to_attr:
        f.write(','.join([str(i) for i in l]))
        # f.write(','.join(attribute[id]))
        f.write('\n')
    f.close()
    print("Success")
        # print(f'attribute {attr} count: {count}')
check()
