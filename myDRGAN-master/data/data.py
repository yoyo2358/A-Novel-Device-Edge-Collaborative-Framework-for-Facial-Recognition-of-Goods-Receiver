import mydataset
import torch

def initdataloader(args):
    if args.mode == 'trainrandom':
        print('mode is {}'.format(args.mode))
        dataset = mydataset.randomData()
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    if args.mode == 'trainsingle':
        print('mode is {}'.format(args.mode)+'\n'+'file list is {}'.format(args.data_path))
        dataset = mydataset.multiPIE(args.data_path, transform_mode='train_transform')
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    if args.mode == 'trainmulti':
        print('mode is {}'.format(args.mode)+'\n'+'file list is {}'.format(args.data_path))
        if args.batch_size % args.images_perID != 0:  
            print("Please give valid combination of batch_size, images_perID");exit();

        dataset = mydataset.multiPIE(args.data_path, transform_mode='train_transform')
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
#测试的话会选择这条分支
    if args.mode in ['gensingle', 'genmulti', 'idensingle', 'idenmulti']:
        print('mode is {}'.format(args.mode)+'\n'+'file list is {}'.format(args.data_path))
        if args.mode in ['genmulti', 'idenmulti'] and args.batch_size % args.images_perID != 0:  
            print("Please give valid combination of batch_size, images_perID");exit();

        dataset = mydataset.multiPIE(args.data_path, transform_mode='test_transform')
        print("dataset has loaded successfully")
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        print("dataloader has loaded and return successfully")
    return dataloader