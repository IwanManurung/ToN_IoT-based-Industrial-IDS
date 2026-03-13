class params:
    raw_dir = 'data_src/ToN_IoT/'
    raw_edge = 'data_src/ToN_IoT/Processed_IoT_dataset/'
    raw_linux = 'data_src/ToN_IoT/Processed_Linux_dataset/'
    raw_windows = 'data_src/ToN_IoT/Processed_Windows_dataset/'
    fog_dir = 'datastream/ToN_IoT_Fog_dataset/'
    edge_dir = 'datastream/ToN_IoT_Edge_dataset/'
    saved_dir = 'src/saved_model/'

    fog_layer = 'fog'
    edge_layer = 'edge'

    csv_sources = {'edge_groundtruth': f'{edge_dir}Edge_GroundTruth.csv',
                'fridge': f'{edge_dir}Edge_IoT_Fridge.csv',
                'garage_door': f'{edge_dir}Edge_IoT_Garage_Door.csv',
                'modbus': f'{edge_dir}Edge_IoT_Modbus.csv',
                'motion_light': f'{edge_dir}Edge_IoT_Motion_Light.csv',
                'thermostat': f'{edge_dir}Edge_IoT_Thermostat.csv',
                'weather': f'{edge_dir}Edge_IoT_Weather.csv',
                'linux_disks':f'{fog_dir}linux_disks_feature.csv',
                'linux_memory': f'{fog_dir}linux_memory_feature.csv',
                'linux_process': f'{fog_dir}linux_process_feature.csv',
                'linux_groundtruth': f'{fog_dir}Fog_GroundTruth.csv',
                'windows7': f'{fog_dir}windows7.csv',
                'windows10': f'{fog_dir}windows10.csv',
                }

    edge_device_list = ['fridge', 'garage_door', 'modbus', 'motion_light', 'thermostat', 'weather']
    fog_device_list = ['linux', 'windows7', 'windows10']
    
    # npz_sources = {'fog': {'fridge': f'{fog_dir}fog_fridge.npz',
    #                        'garage_door': f'{fog_dir}fog_garage_door.npz',
    #                        'modbus': f'{fog_dir}fog_modbus.npz',
    #                        'motion_light': f'{fog_dir}fog_motion_light.npz',
    #                        'thermostat': f'{fog_dir}fog_thermostat.npz',
    #                        'weather': f'{fog_dir}fog_weather.npz'},
    #                 'edge': {'fridge': f'{edge_dir}Edge_IoT_Fridge.npz',
    #                          'garage_door': f'{edge_dir}Edge_IoT_Garage_Door.npz',
    #                          'modbus': f'{edge_dir}Edge_IoT_Modbus.npz',
    #                          'motion_light': f'{edge_dir}Edge_IoT_Motion_Light.npz',
    #                          'thermostat': f'{edge_dir}Edge_IoT_Thermostat.npz',
    #                          'weather': f'{edge_dir}Edge_IoT_Weather.npz'}
                    # }
    target_label = 'target'
    ts_label = 'ts'
    normal_label = 'normal'
    normal_l1label = 'normal'
    attack_l1label= 'malicious'
    non_feature = [ts_label, target_label]
    saved_truth_table = f'{saved_dir}truth_table.csv'
    rep_None = -99
    feature_naming = {'garage_door': ['door_state', 'sphone_signal'],
                      'weather': ['temperature', 'pressure', 'humidity'],
                      'motion_light': ['motion_status', 'light_status'],
                      'fridge': ['fridge_temperature', 'temp_condition'],
                      'thermostat': ['current_temperature', 'thermostat_status'],
                      'modbus': ['FC1_Read_Input_Register', 'FC2_Read_Discrete_Value',
                                 'FC3_Read_Holding_Register', 'FC4_Read_Coil']
                     }
    edge_fusion_csv_sources = {'weather': f'{edge_dir}Edge_Fusion_Weather.csv',
                               'fridge': f'{edge_dir}Edge_Fusion_Fridge.csv',
                               'thermostat': f'{edge_dir}Edge_Fusion_Thermostat.csv',
                               'motion_light': f'{edge_dir}Edge_Fusion_Motion_Light.csv',
                               'garage_door': f'{edge_dir}Edge_Fusion_Garage_Door.csv',
                               'modbus': f'{edge_dir}Edge_Fusion_Modbus.csv'
                              }
    fog_fusion_csv_sources = {'weather': f'{fog_dir}Fog_Fusion_Weather.csv',
                              'fridge': f'{fog_dir}Fog_Fusion_Fridge.csv',
                              'thermostat': f'{fog_dir}Fog_Fusion_Thermostat.csv',
                              'motion_light': f'{fog_dir}Fog_Fusion_Motion_Light.csv',
                              'garage_door': f'{fog_dir}Fog_Fusion_Garage_Door.csv',
                              'modbus': f'{fog_dir}Fog_Fusion_Modbus.csv'
                              }
    iids_model = ["AnomalyModel", "ClassifierModel", "DelayedLabelModel"]