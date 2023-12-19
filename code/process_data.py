import pandas as pd
import numpy as np

def process_shot_group(group):
    minor_event_tracker = 0
    minor_events = []
    #sort by order column
    group = group.sort_values(by=['Order'])

    for i in group.index[::20]:  # Iterate over the index, stepping by 20
        minor_events.extend([minor_event_tracker] * 20)


        if group.at[i, 'Label'] == 1:  # Check the label at every 20th row
            minor_event_tracker += 1
        elif group.at[i, 'Label'] == 2:
            minor_event_tracker +=5

    group['Minor events that occurred yet'] = minor_events
    return group

def count_minor_events(df_data):
    #make a for loop per shot
    shots = df_data['Shot'].unique()
    #add a column to the dataframe called order
    df_data['Order'] = 0
    for shot in shots :
        #take frame 0
        df_shot = df_data[df_data['Shot'] == shot]
        df_shot = df_shot[df_shot['Frame'] == 0]
        #take the time of the frame 0
        time = df_shot['Time'].values

        for l in range (len(time)):
            #take the index in the data of the smallest time
            index = df_shot['Time'].idxmin()
            #set the order column of the index to 1
            df_data.loc[index, 'Order'] = l
            #set the order to the next 18 rows
            for i in range(1, 20):
                df_data.loc[index+i, 'Order'] = l
            #drop the value of the time array so the next smallest value will be taken
            df_shot = df_shot.drop(index)

    # Apply the function to each 'Shot' group
    grouped_df = df_data.groupby('Shot').apply(process_shot_group).reset_index(drop=True)
    # Display the result
    grouped_df = grouped_df.groupby(by=['Shot', 'Order'], group_keys=False).apply(lambda x: x.sort_values(by=['Frame'])).reset_index(drop=True)

    grouped_df['Label'] = grouped_df['Label'].astype(float)
    return grouped_df

def data_to_window(df_data):
    #get the distance from the each row
    #create a dictionnary with a distance column
    dict_windows = {'distance_mean':[],'distances':[],'ECEcore' :[],'ECEcore_mean':[],'ECEcore_dx1':[],'ECEcore_dx2':[], 'ZMAG':[],'ZMAG_mean' :[],'LI' :[], 'LI_mean' :[],'IPLA' :[], 'IPLA_dx1' :[], 'IPLA_dx2' : [], 'IPLA_dx3' :[], 'IPLA_ddx' : [], 'fft_low': [],'IPLA_1' : [], 'IPLA_2' : [], 'IPLA_3' :[], 'IPLA_4' : [],'IPLA_mean':[],'time_window' : [], 'Vloop_1' : [],'Vloop_2' : [], 'Vloop_3' :[], 'Vloop_4' :[]}

    df_data_sliced = df_data

    df_data_distance = df_data_sliced['Distance']
    df_data_distance = df_data_distance.to_numpy().astype(np.float64)
    #get the distance from the each row
    distance_mean = np.mean(df_data_distance)

    dict_windows['distances'] = (df_data_distance)
    dict_windows['distance_mean'] = (distance_mean)

    df_data_current = df_data_sliced['IPLA']
    df_data_current = df_data_current.to_numpy()

    #take the fft of the current
    fft_res = np.fft.fft(df_data_current)
    fft_res = np.abs(np.array(fft_res))
    fft_low = fft_res[0]
    dict_windows['fft_low'] = fft_low

    #A CHANGER POUR LES VRAIES WINDOWS
    dict_windows['IPLA']=(df_data_current)
    dict_windows['IPLA_1']=(df_data_current[14])
    dict_windows['IPLA_2']=(df_data_current[6])
    dict_windows['IPLA_3']=(df_data_current[7])
    dict_windows['IPLA_4']=(df_data_current[5])
    dict_windows['IPLA_mean']=(np.mean(df_data_current))

    dict_windows['Vloop_1']=(df_data_sliced['Vloop'].to_numpy()[14])
    dict_windows['Vloop_2']=((df_data_sliced['Vloop'].to_numpy()[13]+df_data_sliced['Vloop'].to_numpy()[11]+df_data_sliced['Vloop'].to_numpy()[10])/3)
    dict_windows['Vloop_3']=(((df_data_sliced['Vloop'].to_numpy()[5]+df_data_sliced['Vloop'].to_numpy()[6]+df_data_sliced['Vloop'].to_numpy()[7]+df_data_sliced['Vloop'].to_numpy()[8])/4))
    dict_windows['Vloop_4']=(df_data_sliced['Vloop'].to_numpy()[2])

    df_data_ECEcore = df_data_sliced['ECEcore'].to_numpy()

    dict_windows['ECEcore']=(df_data_ECEcore)
    dict_windows['ECEcore_mean'] = np.mean(df_data_ECEcore)
    dict_windows['ECEcore_dx1']=(np.diff(df_data_ECEcore)[-2:][0])
    dict_windows['ECEcore_dx2']=(np.diff(df_data_ECEcore)[-2:][1])

        
   
    df_data_ZMAG = df_data_sliced['ZMAG'].to_numpy()
    dict_windows['ZMAG']=(df_data_ZMAG)
    dict_windows['ZMAG_mean']=(np.mean(df_data_ZMAG))

    df_data_LI = df_data_sliced['LI'].to_numpy()
    dict_windows['LI']=(df_data_LI)
    dict_windows['LI_mean']=(np.mean(df_data_LI))

    #dict_windows['window']=(window_j)
    dict_windows['time_window'] = df_data_sliced['Time'].to_numpy()[0]

    
    #take the derivative of the distance
    df_data_derivative = np.diff(df_data_current)[-3:]
    dict_windows['IPLA_dx1']=(df_data_derivative[0])
    dict_windows['IPLA_dx2']=(df_data_derivative[1])
    dict_windows['IPLA_dx3']=(df_data_derivative[2])

    #take the double derivative of the distance
    df_data_IPLA_ddx = np.diff(df_data_derivative)[-1:]
    dict_windows['IPLA_ddx']=(df_data_IPLA_ddx)

 
    #remove the keys distances and IPLA, ZMAG, ECEcore, LI
    dict_windows.pop('distances')
    dict_windows.pop('IPLA')
    dict_windows.pop('ZMAG')
    dict_windows.pop('ECEcore')
    dict_windows.pop('LI')
    

    #dict_windows to dataframe
    df_windows = pd.DataFrame.from_dict(dict_windows)


    return df_windows


 

def NN_data_to_window(df_data):
    #get the distance from the each row
    #create a dictionnary with a distance column
    dict_windows = {'distance_mean':[],'distances':[],'ECEcore' :[],'Instability' :[],'ECEcore_mean':[],'ECEcore_dx1':[],'ECEcore_dx2':[], 'ZMAG':[],'ZMAG_mean' :[],'LI' :[], 'LI_mean' :[],'IPLA' :[], 'IPLA_dx1' :[], 'IPLA_dx2' : [], 'IPLA_dx3' :[], 'IPLA_ddx' : [], 'fft_low': [],'IPLA_1' : [], 'IPLA_2' : [], 'IPLA_3' :[], 'IPLA_4' : [],'IPLA_mean':[],'time_window' : [], 'Vloop_1' : [],'Vloop_2' : [], 'Vloop_3' :[], 'Vloop_4' :[]}

    df_data_sliced = df_data

    df_data_distance = df_data_sliced['IPE']
    df_data_distance = df_data_distance.to_numpy().astype(np.float64)
    #get the distance from the each row
    distance_mean = np.mean(df_data_distance)

    dict_windows['distances'] = (df_data_distance)
    dict_windows['distance_mean'] = (distance_mean)

    df_data_current = df_data_sliced['IPLA']
    df_data_current = df_data_current.to_numpy()

    #take the fft of the current
    fft_res = np.fft.fft(df_data_current)
    fft_res = np.abs(np.array(fft_res))
    fft_low = fft_res[0]
    dict_windows['fft_low'] = fft_low

    #A CHANGER POUR LES VRAIES WINDOWS
    dict_windows['IPLA']=(df_data_current)
    dict_windows['IPLA_1']=(df_data_current[14])
    dict_windows['IPLA_2']=(df_data_current[6])
    dict_windows['IPLA_3']=(df_data_current[7])
    dict_windows['IPLA_4']=(df_data_current[5])
    dict_windows['IPLA_mean']=(np.mean(df_data_current))

    dict_windows['Vloop_1']=(df_data_sliced['Vloop'].to_numpy()[14])
    dict_windows['Vloop_2']=((df_data_sliced['Vloop'].to_numpy()[13]+df_data_sliced['Vloop'].to_numpy()[11]+df_data_sliced['Vloop'].to_numpy()[10])/3)
    dict_windows['Vloop_3']=(((df_data_sliced['Vloop'].to_numpy()[5]+df_data_sliced['Vloop'].to_numpy()[6]+df_data_sliced['Vloop'].to_numpy()[7]+df_data_sliced['Vloop'].to_numpy()[8])/4))
    dict_windows['Vloop_4']=(df_data_sliced['Vloop'].to_numpy()[2])

    df_data_ECEcore = df_data_sliced['ECEcore'].to_numpy()

    dict_windows['ECEcore']=(df_data_ECEcore)
    dict_windows['ECEcore_mean'] = np.mean(df_data_ECEcore)
    dict_windows['ECEcore_dx1']=(np.diff(df_data_ECEcore)[-2:][0])
    dict_windows['ECEcore_dx2']=(np.diff(df_data_ECEcore)[-2:][1])

        
   
    df_data_ZMAG = df_data_sliced['ZMAG'].to_numpy()
    dict_windows['ZMAG']=(df_data_ZMAG)
    dict_windows['ZMAG_mean']=(np.mean(df_data_ZMAG))

    df_data_LI = df_data_sliced['LI'].to_numpy()
    dict_windows['LI']=(df_data_LI)
    dict_windows['LI_mean']=(np.mean(df_data_LI))

    #dict_windows['window']=(window_j)
    dict_windows['time_window'] = df_data_sliced['Time'].to_numpy()[0]

    dict_windows['Instability'] = df_data_sliced['Instability'].to_numpy()[0]
    
    #take the derivative of the distance
    df_data_derivative = np.diff(df_data_current)[-3:]
    dict_windows['IPLA_dx1']=(df_data_derivative[0])
    dict_windows['IPLA_dx2']=(df_data_derivative[1])
    dict_windows['IPLA_dx3']=(df_data_derivative[2])

    #take the double derivative of the distance
    df_data_IPLA_ddx = np.diff(df_data_derivative)[-1:]
    dict_windows['IPLA_ddx']=(df_data_IPLA_ddx)

 
    #remove the keys distances and IPLA, ZMAG, ECEcore, LI
    dict_windows.pop('distances')
    dict_windows.pop('IPLA')
    dict_windows.pop('ZMAG')
    dict_windows.pop('ECEcore')
    dict_windows.pop('LI')
    

    #dict_windows to dataframe
    df_windows = pd.DataFrame.from_dict(dict_windows)


    return df_windows


 
