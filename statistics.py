from statistics_jinnian import save_plot
import numpy as np

def line_analysis(line):
    if 'lr' in line and 'loss' in line:
        epoch = int(line.split('/300]',1)[0].rsplit('[',1)[1])
        #print(line.split('lr ',1))
        #print(line.split('lr ',1)[1].split('\t',1))
        lr = float(line.split('lr ',1)[1].split('\t',1)[0])
        step = int(line.split('/300][', 1)[1].split('/',1)[0])
        loss = float(line.split('loss ', 1)[1].split(' ', 1)[0])
        return epoch, step, lr, loss
    else:
        return None, None, None, None

def main():
    log_name = 'logs/Swin-T-train-log-reference.txt'
    def get_train_info(log_name):
        train_info = {'loss': {}, 'lr': {}} # loss {'epoch', 'step'}, learning rate
        with open(log_name, "r") as f:
            for line in f:
                epoch, step, lr, loss = line_analysis(line)
                if epoch is not None:
                    if epoch not in train_info['loss']:
                        train_info['loss'][epoch] = [loss]
                        train_info['lr'][epoch] = [lr]
                    else:
                        train_info['loss'][epoch].append(loss)
                        train_info['lr'][epoch].append(lr)
        return train_info
    #print(len(train_info['loss']), train_info['loss'].keys())
    #print(len(train_info['lr']), train_info['lr'].keys())

    #train_info_org = get_train_info(log_name)
    #train_info_mine = get_train_info('logs/log_rank0.txt')
    train_info_org = get_train_info('logs/log_inter_1e-4.txt')
    train_info_mine = get_train_info('logs/log_inter_1e-6.txt')
    x_axis = list(train_info_org['loss'].keys())
    loss_org = np.array([np.mean(train_info_org['loss'][epoch]) for epoch in x_axis])
    lr_org = np.array([train_info_org['lr'][epoch][0] for epoch in x_axis])
    x_mine_axis = list(train_info_mine['loss'].keys())
    loss_mine = np.array([np.mean(train_info_mine['loss'][epoch]) for epoch in x_mine_axis])
    #loss_1 = []
    #loss_2 = []
    #for epoch in x_axis
    lr_mine = [train_info_mine['lr'][epoch][0] for epoch in x_mine_axis]
    lr_mine = np.array([0]*72 + lr_mine)
    #print(len(train_info_mine['loss']), train_info_mine['loss'].keys())
    #print(len(train_info_mine['lr']), train_info_mine['lr'].keys())
    #save_plot(x_axis, np.array(loss)[:,np.newaxis], 'loss_org.png', transparent=False, marker_color_list=['b-'])
    #save_plot(x_axis, np.stack([lr_org, lr_mine], axis=1), 'lr.png', transparent=False, marker_color_list=['b-', 'r-'], legend=['lr_org', 'lr_mine'])
    save_plot(x_axis, np.stack([loss_org, loss_mine], axis=1), 'loss.png', transparent=False, marker_color_list=['b-', 'r-'], legend=['loss_1e-4', 'loss_1e-6'])


if __name__ == '__main__':
    main()