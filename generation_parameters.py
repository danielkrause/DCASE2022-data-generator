# Parameters used in the data generation process.


def get_params(argv='1'):
    print("SET: {}".format(argv))
    # ########### default parameters (NIGENS data) ##############
    params = dict(
        db_name = 'nigens',
        rirpath = 'E:/DCASE2022/TAU_Spatial_RIR_Database_2021/RIR_DB',
        mixturepath = 'E:/DCASE2022/TAU_Spatial_RIR_Database_2021/Dataset-NIGENS',
        noisepath = 'E:/DCASE2022/TAU_Spatial_RIR_Database_2021/Noise_DB',
        nb_folds = 2,
        rooms2fold = [[10, 6, 1, 4, 3, 8], # FOLD 1
                      [9, 5, 2, 0, 0, 0]], # FOLD 2
        db_path = 'E:/DCASE2022/TAU_Spatial_RIR_Database_2021/Code/NIGENS',
        max_polyphony = 2,
        active_classes = [0, 1, 2, 3, 5, 6, 8, 9, 10, 11, 12, 13],
        nb_mixtures_per_fold = [600, 300], # if scalar, same number of mixtures for each fold
        mixture_duration = 60., #in seconds
        event_time_per_layer = 40., #in seconds (should be less than mixture_duration)
            )
        

    # ########### User defined parameters ##############
    if argv == '1':
        print("USING DEFAULT PARAMETERS FOR NIGENS DATA\n")

    elif argv == '2': ###### FSD50k DATA
        params['db_name'] = 'fsd50k'
        params['db_path']= 'E:/DCASE2022/TAU_Spatial_RIR_Database_2021/Code/FSD50K'
        params['mixturepath'] = 'E:/DCASE2022/TAU_Spatial_RIR_Database_2021/Dataset-FSD'
        params['active_classes'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    else:
        print('ERROR: unknown argument {}'.format(argv))
        exit()

    for key, value in params.items():
        print("\t{}: {}".format(key, value))
    return params