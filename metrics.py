## For ani

## again 2 sympy in

import pandas as pd



plays = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2024/plays.csv') # plays_df
print('plays :', plays.shape)

tackles = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2024/tackles.csv')
print('tackles :', tackles.shape)

plays_tackles_merged = pd.merge(plays,tackles,on=['gameId','playId'],how='inner')
print('plays_tackles_merged :',plays_tackles_merged.shape)

    
    
indicator = {}
indicators = {}

indicators_animation = {}
t_nt = []

for week in range(1,10):
    
    print()
    print('--------------->') 
    print()


    week_i = week
    tracking_week_i = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2024/tracking_week_'+str(week_i)+'.csv')
    print('tracking_week_'+str(week_i), tracking_week_i.shape)
    
    plays_tackles_tracking_i_merged = pd.merge(plays_tackles_merged,tracking_week_i,on=['gameId','playId'],how='inner')
    print('plays_tackles_tracking_i_merged', plays_tackles_tracking_i_merged.shape)     
    
    
    for q in range(1,5) :
    
        quarter_j = q
        plays_tackles_tracking_i_merged_quarter_j = plays_tackles_tracking_i_merged[plays_tackles_tracking_i_merged['quarter']==quarter_j]
        print('quarter_j and plays_tackles_tracking_i_merged_quarter_j.shape:', quarter_j, plays_tackles_tracking_i_merged_quarter_j.shape )
    
    
        dtype_dict = {'passResult':'str', 'foulName1':'str', 'foulName2': 'str'}
        plays_tackles_tracking_i_merged_quarter_j = plays_tackles_tracking_i_merged_quarter_j.astype(dtype_dict)
        print('week_i = ', week_i,'quarter_j = ', quarter_j)

        plays_tackles_tracking_i_merged_quarter_j_defence = plays_tackles_tracking_i_merged_quarter_j[plays_tackles_tracking_i_merged_quarter_j['nflId_x'] == plays_tackles_tracking_i_merged_quarter_j['nflId_y']]
        print('plays_tackles_tracking_i_merged_quarter_j_defence.shape',plays_tackles_tracking_i_merged_quarter_j_defence.shape)
        
        plays_tackles_tracking_i_merged_quarter_j_offence = plays_tackles_tracking_i_merged_quarter_j[(plays_tackles_tracking_i_merged_quarter_j['nflId_y'] == plays_tackles_tracking_i_merged_quarter_j['ballCarrierId'])]
        print('plays_tackles_tracking_i_merged_quarter_j_offence.shape', plays_tackles_tracking_i_merged_quarter_j_offence.shape)

        ## tackle and nontackle data  tac nt

        plays_tackles_tracking_i_merged_quarter_j_defence_tackle = plays_tackles_tracking_i_merged_quarter_j_defence[(plays_tackles_tracking_i_merged_quarter_j_defence['tackle'] == 1)]
        plays_tackles_tracking_i_merged_quarter_j_defence_nt = plays_tackles_tracking_i_merged_quarter_j_defence[(plays_tackles_tracking_i_merged_quarter_j_defence['tackle'] == 0)]

        plays_tackles_tracking_i_merged_quarter_j_offence_tackle = plays_tackles_tracking_i_merged_quarter_j_offence[(plays_tackles_tracking_i_merged_quarter_j_offence['tackle'] == 1)]
        plays_tackles_tracking_i_merged_quarter_j_offence_nt = plays_tackles_tracking_i_merged_quarter_j_offence[(plays_tackles_tracking_i_merged_quarter_j_offence['tackle'] == 0)]

        print('plays_tackles_tracking_i_merged_quarter_j_defence_(tackle/nt/t+nt).shapes', plays_tackles_tracking_i_merged_quarter_j_defence_tackle.shape,plays_tackles_tracking_i_merged_quarter_j_defence_nt.shape,plays_tackles_tracking_i_merged_quarter_j_defence_tackle.shape[0]+plays_tackles_tracking_i_merged_quarter_j_defence_nt.shape[0])
        print('plays_tackles_tracking_i_merged_quarter_j_offence_(tackle/nt/t+nt).shapes', plays_tackles_tracking_i_merged_quarter_j_offence_tackle.shape,plays_tackles_tracking_i_merged_quarter_j_offence_nt.shape,plays_tackles_tracking_i_merged_quarter_j_offence_tackle.shape[0]+plays_tackles_tracking_i_merged_quarter_j_offence_nt.shape[0])
    
        plays_tackles_tracking_i_merged_quarter_j_defence_tackle_grouped = plays_tackles_tracking_i_merged_quarter_j_defence_tackle.groupby(['gameId', 'playId'])
        plays_tackles_tracking_i_merged_quarter_j_defence_nt_grouped = plays_tackles_tracking_i_merged_quarter_j_defence_nt.groupby(['gameId', 'playId'])

        plays_tackles_tracking_i_merged_quarter_j_offence_tackle_grouped = plays_tackles_tracking_i_merged_quarter_j_offence_tackle.groupby(['gameId', 'playId'])
        plays_tackles_tracking_i_merged_quarter_j_offence_nt_grouped = plays_tackles_tracking_i_merged_quarter_j_offence_nt.groupby(['gameId', 'playId'])

        print('ngroups')
        print(plays_tackles_tracking_i_merged_quarter_j_defence_tackle_grouped.ngroups, plays_tackles_tracking_i_merged_quarter_j_defence_nt_grouped.ngroups)
        print(plays_tackles_tracking_i_merged_quarter_j_offence_tackle_grouped.ngroups, plays_tackles_tracking_i_merged_quarter_j_offence_nt_grouped.ngroups)
    
        
        import numpy as np
        import math
        from tqdm import tqdm

        from scipy.optimize import minimize

        
#         import sympy as sp
        def conics(params,x,y):
            a,b,c,d,e,f = params
            return a*x**2 + b*x*y + c*y**2 + d*x + e*y + f

        def objective_fn(params,x,y):
            return np.sum(conics(params,x,y)**2)


        groups = [('plays_tackles_tracking_i_merged_quarter_j_defence_tackle_grouped', plays_tackles_tracking_i_merged_quarter_j_defence_tackle_grouped),
                  ('plays_tackles_tracking_i_merged_quarter_j_defence_nt_grouped', plays_tackles_tracking_i_merged_quarter_j_defence_nt_grouped), 
                  ('plays_tackles_tracking_i_merged_quarter_j_offence_tackle_grouped', plays_tackles_tracking_i_merged_quarter_j_offence_tackle_grouped), 
                  ('plays_tackles_tracking_i_merged_quarter_j_offence_nt_grouped', plays_tackles_tracking_i_merged_quarter_j_offence_nt_grouped)]

#         indicator = {}
#         indicators = {}
        
    
#         gameid_ntac, playid_ntac = 2022110603, 3085
#         gameid_tac, playid_tac = 2022110609, 4162
    
    
#         indicators_ani = {}
        # for group  in groups :
        for group_name, group in groups : # group_name, group_data     
            print(group_name)
            eccs = []
            loss = []
            conics_type = []
            event_last = []
            theta = []
            coeffs = []
            lengths =  []
            accs = [] #added
            dis =  []
            i = 0
            
#             indicators_ani[group_name] = {}
            indicators_animation[group_name] = {}
            for (gameId, playId), group_data in tqdm(group) :
                
                i +=1

                for k in range(len(group_data['event'])-1,0,-1): 
                    if(str(group_data['event'].iloc[k])!='nan'):

                        lengths.append(len(group_data))

                        l=80 
                        if len(group_data) >= l :# 60 
                            x= group_data['x'][-l:k+1].tolist()  
                            y= group_data['y'][-l:k+1].tolist()
                            s = group_data['s'][-l:k+1].tolist()  
                            acc = group_data['a'][-l:k+1].tolist()
                            o = group_data['o'][-l:k+1].tolist()  
                            direc = group_data['dir'][-l:k+1].tolist()
#                             alined_direc = direc[1:]
                            cos_direc, sin_direc = np.cos(direc), np.sin(direc)
                            sx,sy = s*cos_direc, s*sin_direc
                            lengths.append(len(group_data))

                            dis = group_data['dis'][-l:k+1].tolist()
                            alined_direc = direc[1:]
                        else :
                            x= group_data['x'][5:k+1].tolist() 
                            y= group_data['y'][5:k+1].tolist()
                            s = group_data['s'][5:k+1].tolist()
                            acc = group_data['a'][5:k+1].tolist()
                            o = group_data['o'][5:k+1].tolist()  
                            direc = group_data['dir'][5:k+1].tolist()
                            cos_direc, sin_direc = np.cos(direc), np.sin(direc)
                            sx,sy = s*cos_direc, s*sin_direc
                            dis = group_data['dis'][5:k+1].tolist()
                            aligned_direc = direc[1:]

                        break 


                x_pos = np.array(x) 
                y_pos = np.array(y)
                s = np.array(s)
                acc = np.array(acc)
                o = np.array(o)
                direc = np.array(direc)
                aligned_direc = np.array(direc[1:]) #[1:]
                
                dx, dy = np.diff(x_pos), np.diff(y_pos)
                
    
    
                # init_guess = [1,1,1,1,1,1]
                p0 = [1,1,1,1,1,1]

                params = minimize(objective_fn, p0, args=(x_pos,y_pos))
                params_v = minimize(objective_fn, p0, args=(sx,sy))
                a,b,c,d,e,f = params.x
                coeffs.append((a,b,c,d,e,f))
            
                a_v,b_v,c_v,d_v,e_v,f_v = params_v.x
            
                
                
                # define the symbols
                x, y = sp.symbols('x y')
                x_prime, y_prime = sp.symbols('x_prime y_prime')
                x_new, y_new = sp.symbols('x_new y_new')
                
                theta_ = np.arctan2(b,a-c)/2
                cos_theta_, sin_theta_ = np.cos(theta_), np.sin(theta_)
                
                a_prime = a*cos_theta_**2 +  b*cos_theta_*sin_theta_ +  c*sin_theta_**2
                b_prime = 2*(c-a)*cos_theta_*sin_theta_ + b*(cos_theta_**2 -sin_theta_**2)
                c_prime = a*sin_theta_**2 - b*cos_theta_*sin_theta_ + c*cos_theta_**2
                d_prime = d*cos_theta_ + e*sin_theta_
                e_prime = -d*sin_theta_ + e*cos_theta_
                f_prime =  f
                
                x_coeff_std, x_cen, y_coeff_std, y_cen, const_std = a_prime, d_prime/(2*a_prime), c_prime, e_prime/(2*c_prime), (-f_prime + a_prime*(d_prime/(2*a_prime))**2 + c_prime*(e_prime/(2*c_prime))**2)
                a_check, c_check = const_std/x_coeff_std, const_std/y_coeff_std 
                
                a_prime_v = a*cos_theta_**2 +  b*cos_theta_*sin_theta_ +  c*sin_theta_**2
                b_prime = 2*(c-a)*cos_theta_*sin_theta_ + b*(cos_theta_**2 -sin_theta_**2)
                c_prime_v = a_v*sin_theta_**2 - b*cos_theta_*sin_theta_ + c*cos_theta_**2
                d_prime = d*cos_theta_ + e*sin_theta_
                e_prime = -d*sin_theta_ + e*cos_theta_
                f_prime =  f
                x_coeff_std_v, x_cen_v, y_coeff_std_v, y_cen_v, const_std_v = a_prime, d_prime/(2*a_prime), c_prime, e_prime/(2*c_prime), (-f_prime + a_prime*(d_prime/(2*a_prime))**2 + c_prime*(e_prime/(2*c_prime))**2)
                a_check, c_check = const_std/x_coeff_std, const_std/y_coeff_std 
                
                a_prime_,b_prime_, c_prime_,d_prime_,e_prime_,f_prime_ = sp.symbols('a_prime_,b_prime_, c_prime_,d_prime_,e_prime_,f_prime_')

                D = b**2 -4*a*c
                
                cost = np.cos(np.arctan2(b,a-c)/2)
                sint = np.sin(np.arctan2(b,a-c)/2)
                theta.append(np.arctan2(b,a-c)/2)

                accs.append(np.mean(acc)) 
                # Metric : KC Index 
                eccs.append(   (  (((a_prime*c_prime))) *np.cos(np.mean(direc))*np.mean(s)*np.mean(acc)*((len(x_pos)*.01)**3) *np.cos(np.mean(o)) ) )
                indicators_animation[group_name][(gameId, playId)] = (  (((a_prime*c_prime))) *np.cos(np.mean(direc))*np.mean(s)*np.mean(acc)*((len(x_pos)*.01)**3) *np.cos(np.mean(o)) )
        

            indicator[(group,'eccs')] = eccs
            indicator[(group,'coeffs')] = coeffs
            indicator[(group,'angle')] = theta
            indicator[(group,'lengths')] = lengths
                        
        eccs_pd_dict = {}
        for group_name, group_data in groups :
            eccs_pd = pd.Series(indicator[group_data, 'eccs'])
            eccs_pd_dict[group_name] = eccs_pd.mean() 
        

        eccs_pd_dict

        print('tackle')
        print(eccs_pd_dict['plays_tackles_tracking_i_merged_quarter_j_defence_tackle_grouped']/eccs_pd_dict['plays_tackles_tracking_i_merged_quarter_j_offence_tackle_grouped'])
        print('no_tackle')
        print(eccs_pd_dict['plays_tackles_tracking_i_merged_quarter_j_defence_nt_grouped']/eccs_pd_dict['plays_tackles_tracking_i_merged_quarter_j_offence_nt_grouped'])

        print()
        
        t = eccs_pd_dict['plays_tackles_tracking_i_merged_quarter_j_defence_tackle_grouped']-eccs_pd_dict['plays_tackles_tracking_i_merged_quarter_j_offence_tackle_grouped']
        nt = eccs_pd_dict['plays_tackles_tracking_i_merged_quarter_j_defence_nt_grouped']-eccs_pd_dict['plays_tackles_tracking_i_merged_quarter_j_offence_nt_grouped']
        
        ## Metric : KC Indicator 
        diff = t / nt
        t_nt.append((t,nt))
        
        # print('tackle - no_tackle', diff)
        print('-*-------*-')
        print('diff t-nt', diff)
        # print('diff nt-t', nt-t)
        indicators[(week_i, quarter_j)] = diff
        print('-*-------*-')

        
#         indicators[('week_i', ' quarter_j')] : diff
#         indicators[(, ' quarter_j')] = diff
        indicators[(week_i, quarter_j)] = diff

        print()
    
    
    
    
    
    print()
    print('<<<<<<<<<<<<')    
    
    print()
    
    
