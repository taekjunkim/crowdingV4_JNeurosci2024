#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions used in make_figures.ipynb

## get_SDF
: get spike density function (spike/s) from spike trains

## ANOVA_shape_selectivity
: input: units_df, stimcond 
: ANOVA F-val, P-val from 8 target shapes in a stim_condition (e.g., grCe = gray target alone)

## compute_mod_idx
: modulation index = CS / (C + S - B) - 1; 
: CS = repsonse to target-distractor configuration
: C = repsonse to center target stimulus alone
: S = repsonse to surround distractors alone
: B = Baseline (no stimulus)

## compute_shape_tuning_r2er_n2n
: r2er = unbiased estimate of r2 between two sets of noisy neural responses

@author: Taekjun Kim
"""

import numpy as np; 
from scipy import stats; 
import pandas as pd; 
from utils import er_est as er; 
import matplotlib.pyplot as plt; 

def create_mod_df(crowd_units, crowd_trials, file_dir):
    mod_df = pd.DataFrame(); 
    mod_df['unitID'] = np.arange(len(crowd_units)); 

    mod_df['grCe_grSu18Fa'] = compute_mod_idx(crowd_units, crowd_trials, 'grCe_grSu18Fa'); 
    mod_df['grCe_grSu12Mi'] = compute_mod_idx(crowd_units, crowd_trials, 'grCe_grSu12Mi'); 
    mod_df['grCe_grSu6Ne'] = compute_mod_idx(crowd_units, crowd_trials, 'grCe_grSu6Ne'); 
    mod_df['grCe_grSu1Ne'] = compute_mod_idx(crowd_units, crowd_trials, 'grCe_grSu1Ne'); 
    mod_df['grCe_grSu3Ne'] = compute_mod_idx(crowd_units, crowd_trials, 'grCe_grSu3Ne'); 
    mod_df['coCe_grSu6Ne'] = compute_mod_idx(crowd_units, crowd_trials, 'coCe_grSu6Ne'); 
    mod_df['grCe_grSu12SmCiNe'] = compute_mod_idx(crowd_units, crowd_trials, 'grCe_grSu12SmCiNe'); 
    mod_df['grCe_grSu6CiNe'] = compute_mod_idx(crowd_units, crowd_trials, 'grCe_grSu6CiNe'); 
    mod_df.to_csv(file_dir + 'mod_df.csv'); 
    return mod_df;     

def create_r_er_df(crowd_units, crowd_trials, file_dir):
    r_er_df = pd.DataFrame(); 
    r_er_df['unitID'] = np.arange(len(crowd_units)); 

    df_now = compute_shape_tuning_r2er_n2n(crowd_units, crowd_trials, 'grCe_grSu18Fa'); 
    r_er_df['grCe_grSu18Fa: r_er'] = df_now['r_er']
    r_er_df['grCe_grSu18Fa: r2er'] = df_now['r2er']
    r_er_df['grCe_grSu18Fa: r2er_p'] = df_now['r2er_pval']
    r_er_df['grCe_grSu18Fa: r'] = df_now['rval']
    r_er_df['grCe_grSu18Fa: p'] = df_now['pval']
    del df_now; 

    df_now = compute_shape_tuning_r2er_n2n(crowd_units, crowd_trials, 'grCe_grSu12Mi'); 
    r_er_df['grCe_grSu12Mi: r_er'] = df_now['r_er']
    r_er_df['grCe_grSu12Mi: r2er'] = df_now['r2er']
    r_er_df['grCe_grSu12Mi: r2er_p'] = df_now['r2er_pval']
    r_er_df['grCe_grSu12Mi: r'] = df_now['rval']
    r_er_df['grCe_grSu12Mi: p'] = df_now['pval']
    del df_now; 

    df_now = compute_shape_tuning_r2er_n2n(crowd_units, crowd_trials, 'grCe_grSu6Ne'); 
    r_er_df['grCe_grSu6Ne: r_er'] = df_now['r_er']
    r_er_df['grCe_grSu6Ne: r2er'] = df_now['r2er']
    r_er_df['grCe_grSu6Ne: r2er_p'] = df_now['r2er_pval']
    r_er_df['grCe_grSu6Ne: r'] = df_now['rval']
    r_er_df['grCe_grSu6Ne: p'] = df_now['pval']
    del df_now; 

    df_now = compute_shape_tuning_r2er_n2n(crowd_units, crowd_trials, 'grCe_grSu1Ne'); 
    r_er_df['grCe_grSu1Ne: r_er'] = df_now['r_er']
    r_er_df['grCe_grSu1Ne: r2er'] = df_now['r2er']
    r_er_df['grCe_grSu1Ne: r2er_p'] = df_now['r2er_pval']
    r_er_df['grCe_grSu1Ne: r'] = df_now['rval']
    r_er_df['grCe_grSu1Ne: p'] = df_now['pval']
    del df_now; 

    df_now = compute_shape_tuning_r2er_n2n(crowd_units, crowd_trials, 'grCe_grSu3Ne'); 
    r_er_df['grCe_grSu3Ne: r_er'] = df_now['r_er']
    r_er_df['grCe_grSu3Ne: r2er'] = df_now['r2er']
    r_er_df['grCe_grSu3Ne: r2er_p'] = df_now['r2er_pval']
    r_er_df['grCe_grSu3Ne: r'] = df_now['rval']
    r_er_df['grCe_grSu3Ne: p'] = df_now['pval']
    del df_now; 

    df_now = compute_shape_tuning_r2er_n2n(crowd_units, crowd_trials, 'coCe_grSu6Ne'); 
    r_er_df['coCe_grSu6Ne: r_er'] = df_now['r_er']
    r_er_df['coCe_grSu6Ne: r2er'] = df_now['r2er']
    r_er_df['coCe_grSu6Ne: r2er_p'] = df_now['r2er_pval']
    r_er_df['coCe_grSu6Ne: r'] = df_now['rval']
    r_er_df['coCe_grSu6Ne: p'] = df_now['pval']
    del df_now; 

    df_now = compute_shape_tuning_r2er_n2n(crowd_units, crowd_trials, 'grCe_grSu12SmCiNe'); 
    r_er_df['grCe_grSu12SmCiNe: r_er'] = df_now['r_er']
    r_er_df['grCe_grSu12SmCiNe: r2er'] = df_now['r2er']
    r_er_df['grCe_grSu12SmCiNe: r2er_p'] = df_now['r2er_pval']
    r_er_df['grCe_grSu12SmCiNe: r'] = df_now['rval']
    r_er_df['grCe_grSu12SmCiNe: p'] = df_now['pval']
    del df_now; 

    df_now = compute_shape_tuning_r2er_n2n(crowd_units, crowd_trials, 'grCe_grSu6CiNe'); 
    r_er_df['grCe_grSu6CiNe: r_er'] = df_now['r_er']
    r_er_df['grCe_grSu6CiNe: r2er'] = df_now['r2er']
    r_er_df['grCe_grSu6CiNe: r2er_p'] = df_now['r2er_pval']
    r_er_df['grCe_grSu6CiNe: r'] = df_now['rval']
    r_er_df['grCe_grSu6CiNe: p'] = df_now['pval']
    del df_now; 
    r_er_df.to_csv(file_dir + 'r_er_df.csv')    

    return r_er_df; 

def compute_TC_shape_modulation(crowd_units, crowd_trials, file_dir):

    TC_shape_mod = dict(); 
    stim_conds = [
        'grCe', 'grCe_grSu18Fa', 'grCe_grSu12Mi', 'grCe_grSu6Ne','grCe_grSu1Ne', 
        'grCe_grSu3Ne', 'coCe', 'coCe_grSu6Ne', 'grCe_grSu12SmCiNe', 'grCe_grSu6CiNe'
    ]; 
    TC_shape_mod['unitID'] = crowd_units['unitID'].values; 
    for stimcond in stim_conds:
        TC_shape_mod[f'{stimcond}_PSTH_good'] = np.nan * np.ones((len(crowd_units),600)); 
        TC_shape_mod[f'{stimcond}_PSTH_bad'] = np.nan * np.ones((len(crowd_units),600)); 
        TC_shape_mod[f'{stimcond}_SigDiff'] = np.nan * np.ones((len(crowd_units),600)); 

    for u in np.arange(len(crowd_units)):
        sesID = crowd_units.loc[u,'sesID']; 
        ses_trials = crowd_trials[crowd_trials['sesID']==sesID].reset_index(); 

        for stimcond in stim_conds: 
            if len(np.where(ses_trials['stim_cond']==stimcond)[0])==0:
                continue; 

            # shape ranking
            targ_resp = []; 
            for r in np.arange(8): 
                tNums = np.where((ses_trials['stim_cond']=='grCe') & (ses_trials['target_rot']==r))[0]; 
                respNow = np.mean(np.sum(crowd_units.loc[u,'spkMtx'][tNums,300:700],axis=1)*1000/400,axis=0); 
                respNow2 = 0; 
                # for color condition, shape preference is determined by gray, color target together
                if stimcond[:2]=='co':
                    tNums = np.where((ses_trials['stim_cond']=='coCe') & (ses_trials['target_rot']==r))[0]; 
                    respNow2 = np.mean(np.sum(crowd_units.loc[u,'spkMtx'][tNums,300:700],axis=1)*1000/400,axis=0); 
                targ_resp.append(respNow+respNow2); 
            shape_rank = np.array(targ_resp).argsort()[::-1]; # from best to worst

            # PSTH_good vs. PSTH_bad
            pref_Mtx = np.zeros((0,630)); 
            npref_Mtx = np.zeros((0,630)); 
            for r in np.arange(8):
                rid = shape_rank[r]; 
                tNums = np.where((ses_trials['stim_cond']==stimcond) & (ses_trials['target_rot']==rid))[0]; 
                sdfNow = getSDF(crowd_units.loc[u,'spkMtx'][tNums,185:815],1000); 
                if r<4:
                    pref_Mtx = np.vstack((pref_Mtx,sdfNow)); 
                else:
                    npref_Mtx = np.vstack((npref_Mtx,sdfNow)); 
            TC_shape_mod[f'{stimcond}_PSTH_good'][u,:] = np.mean(pref_Mtx[:,15:615],axis=0); 
            TC_shape_mod[f'{stimcond}_PSTH_bad'][u,:] = np.mean(npref_Mtx[:,15:615],axis=0); 

            # SigDiff
            for t in np.arange(600):
                tStart = t; 
                tEnd = t+30; 
                _, pval = stats.mannwhitneyu(np.mean(pref_Mtx[:,tStart:tEnd],axis=1),np.mean(npref_Mtx[:,tStart:tEnd],axis=1)); 
                if pval<0.05:
                    TC_shape_mod[f'{stimcond}_SigDiff'][u,t] = 1; 
                else:
                    TC_shape_mod[f'{stimcond}_SigDiff'][u,t] = 0; 
        print(f'unitID {u} was done'); 
    
    np.savez(file_dir + "TC_shape_mod.npz", **TC_shape_mod); 
    return TC_shape_mod; 

def getSDF(spk_trn,FS=1000):
    spk_trn = np.array(spk_trn).astype(float)

    # Make gaussian kernel window
    sigma = 5;
    t = np.arange(-3*sigma,3*sigma+1);

    y = (1/sigma*np.sqrt(np.pi*2)) * np.exp(-(t**2)/(2*sigma**2));
    window = y[:];
    window = window/np.sum(window);

    # convolution
    sdf = np.zeros(np.shape(spk_trn));
    for i in np.arange(np.shape(spk_trn)[0]):
        convspike = np.convolve(spk_trn[i,:],window);
        pStart = int(np.floor(len(window)/2));
        pEnd = int(np.floor(len(window)/2)+np.shape(spk_trn)[1]);
        convspike = convspike[pStart:pEnd];
        sdf[i,:] = convspike;
    sdf = sdf*FS;
    return sdf;        

def ANOVA_shape_selectivity(units, trials, stimcond):
    shape_tuning_fval = []; 
    shape_tuning_pval = []; 
    for u in np.arange(len(units)):
        sesID = units.loc[u,'sesID']; 
        ses_trials = trials[trials['sesID']==sesID].reset_index(); 

        # get responses from 8 target shapes
        targ_resp = []; 
        for r in np.arange(8): 
            tNums = np.where((ses_trials['stim_cond']==stimcond) & (ses_trials['target_rot']==r))[0]; 
            respNow = np.sum(units.loc[u,'spkMtx'][tNums,300:700],axis=1)*1000/400; 
            #respNow = np.nanmean(getSDF(units.loc[u,'spkMtx'][tNums,:])[:,300:700],axis=1); 
            targ_resp.append(respNow); 
        
        fval, pval = stats.f_oneway(
            targ_resp[0],targ_resp[1],targ_resp[2],targ_resp[3],
            targ_resp[4],targ_resp[5],targ_resp[6],targ_resp[7]
        ); 
        shape_tuning_fval.append(fval);              
        shape_tuning_pval.append(pval);        
    
    return np.array(shape_tuning_fval), np.array(shape_tuning_pval)

def compute_mod_idx(units, trials, stimcond):
    mod_idx_mtx = []; 
    for u in np.arange(len(units)):
        sesID = units.loc[u,'sesID']; 
        ses_trials = trials[trials['sesID']==sesID].reset_index(); 

        u_bar = stimcond.find('_')
        targ_cond = stimcond[:u_bar]; 
        dist_cond = stimcond[u_bar+1:]; 

        if len(np.where(ses_trials['stim_cond']==dist_cond)[0])==0:
            mod_idx_mtx.append(np.nan); 
            continue; 

        # stimcond
        cond_resp = 0; 
        for r in np.arange(8): 
            tNums = np.where((ses_trials['stim_cond']==stimcond) & (ses_trials['target_rot']==r))[0]; 
            respNow = np.mean(np.sum(units.loc[u,'spkMtx'][tNums,300:700],axis=1)*1000/400,axis=0); 
            cond_resp += respNow; 
        cond_resp = cond_resp/8; 

        # target
        targ_resp = 0; 
        for r in np.arange(8): 
            tNums = np.where((ses_trials['stim_cond']==targ_cond) & (ses_trials['target_rot']==r))[0]; 
            respNow = np.mean(np.sum(units.loc[u,'spkMtx'][tNums,300:700],axis=1)*1000/400,axis=0); 
            targ_resp += respNow; 
        targ_resp = targ_resp/8; 

        # NoStim
        tNums = np.where((ses_trials['stim_cond']=='NoStim'))[0]; 
        NoStim = np.mean(np.sum(units.loc[u,'spkMtx'][tNums,300:700],axis=1)*1000/400,axis=0); 
        
        # distractor
        tNums = np.where((ses_trials['stim_cond']==dist_cond))[0]; 
        dist_resp = np.mean(np.sum(units.loc[u,'spkMtx'][tNums,300:700],axis=1)*1000/400,axis=0); 

        lin_sum = targ_resp + dist_resp - NoStim; 
        mod_idx = (cond_resp/lin_sum) - 1; 
        mod_idx_mtx.append(mod_idx); 

    return np.array(mod_idx_mtx); 
    
def compute_shape_tuning_r2er_n2n(units, trials, stimcond):
    """
    1. compute r2er_n2n
        def r2er_n2n(x,y)
            == Input
            x : numpy.ndarray
                N neurons X n trials X m observations array
            y : numpy.ndarray
                N neurons X n trials X m observations array
            == Returns
            r2er : an estimate of the r2 between the expected values of the data
    2. Run the simulation with r2er=0 but all the other same parameters as estimated from your data and determine the fraction that exceeds your observed r2er
        repeat 1000 times
            x_shuffled: condition shuffled within x
            y_shuffled: condition shuffled within y
            r2er_n2n(x_shuffled, y_shuffled)
        get confidence interval
        compare with r2er_n2n(x,y)
    """
    condA = 'grCe'; 
    condB = stimcond; 

    nUnits = np.shape(units)[0]; 
    tuning_corr_df = pd.DataFrame(columns=['unitID','r2er','r2er_pval','rval','pval'])
    for u in np.arange(nUnits):
        sesID = units.loc[u,'sesID']; 
        ses_t = trials[trials['sesID']==sesID].reset_index(); 
        nTrials = np.max(ses_t['rep_num']); 
        spkMtx = units.loc[u,'spkMtx']; 

        ### check if this stimcond was tested in this session
        if len(np.where(ses_t['stim_cond']==stimcond)[0])==0:
            tuning_corr_df.loc[u,'unitID'] = u; 
            tuning_corr_df.loc[u,'r2er'] = np.nan; 
            tuning_corr_df.loc[u,'r2er_pval'] = np.nan; 
            tuning_corr_df.loc[u,'rval'] = np.nan; 
            tuning_corr_df.loc[u,'pval'] = np.nan;     
            continue; 

        ### organize data into (nTrials, mCondition)
        data_A = np.empty((nTrials, 8)); 
        data_A[:] = np.nan; 
        data_B = np.empty((nTrials, 8)); 
        data_B[:] = np.nan; 

        for m in np.arange(8):                # target rotation (mCondition = 8)
            for n in np.arange(nTrials):      # repetition number (nTrials)
                rowNum_A = np.where(
                    (ses_t['stim_cond']==condA) &
                    (ses_t['target_rot']==m) &                    
                    (ses_t['rep_num']==n)                                        
                )[0]
                rowNum_B = np.where(
                    (ses_t['stim_cond']==condB) &
                    (ses_t['target_rot']==m) &                    
                    (ses_t['rep_num']==n)                                        
                )[0]
                if len(rowNum_A)>0:
                    data_A[n,m] = np.sum(spkMtx[rowNum_A,300:700])*1000/400; 
                if len(rowNum_B)>0:
                    data_B[n,m] = np.sum(spkMtx[rowNum_B,300:700])*1000/400; 

        ### remove nan trial
        valid_A = np.where(np.isnan(np.sum(data_A,axis=1))==0)[0]; 
        data_A = data_A[valid_A,:]; 
        valid_B = np.where(np.isnan(np.sum(data_B,axis=1))==0)[0]; 
        data_B = data_B[valid_B,:];

        ### compute r2er
        r2er, r2 = er.r2er_n2n(data_A, data_B); 
        if r2er[0][0]>1:
            r2er[0][0] = 1; 
        elif r2er[0][0]<0:
            r2er[0][0] = 0; 
        
        ### pearson corr
        rval, pval = stats.pearsonr(np.mean(data_A,axis=0), np.mean(data_B,axis=0)); 
    
        ### bootstrap
        sim_r2er = []; 
        for b in np.arange(1000):
            data_A_bs = data_A.copy();
            data_A_bs = data_A_bs[:,np.random.permutation(8)];   ### condition shuffled
            data_B_bs = data_B.copy();
            data_B_bs = data_B_bs[:,np.random.permutation(8)];   ### condition shuffled
            r2er_sim, r2_sim = er.r2er_n2n(data_A_bs,data_B_bs); 
            if r2er_sim[0][0]>1:
                r2er_sim[0][0] = 1; 
            elif r2er_sim[0][0]<0:
                r2er_sim[0][0] = 0; 
            sim_r2er.append(r2er_sim[0][0]); 
        sim_r2er = np.array(sim_r2er).squeeze();  

        ### add information
        tuning_corr_df.loc[u,'unitID'] = u; 
        tuning_corr_df.loc[u,'r2er'] = r2er[0][0]; 
        if rval>0:
            tuning_corr_df.loc[u,'r_er'] = r2er[0][0]**0.5;             
        else:
            tuning_corr_df.loc[u,'r_er'] = -r2er[0][0]**0.5;                         
        tuning_corr_df.loc[u,'r2er_pval'] = len(np.where(sim_r2er>r2er[0][0])[0])/1000; 
        tuning_corr_df.loc[u,'rval'] = rval; 
        tuning_corr_df.loc[u,'pval'] = pval;     

        if u%10==0:
            print(f'unitID: {u} was done'); 
    return tuning_corr_df; 

def draw_raster(crowd_units, crowd_trials, unitID, conditions, colors):
    stimOn = 300; 
    preTime = 100; 
    postTime = 500; 

    sesID = crowd_units.loc[unitID,'sesID']; 
    ses_t = crowd_trials[crowd_trials['sesID']==sesID].reset_index(); 
    spkMtx = crowd_units.loc[unitID,'spkMtx']; 
    psthMtx = getSDF(spkMtx,1000); 

    ### shape preference ranking
    grCe_psth = np.empty((8,600)); 
    for r in np.arange(8):
        tNow = np.where((ses_t['stim_cond']=='grCe') & (ses_t['target_rot']==r))[0]; 
        grCe_psth[r,:] = np.mean(psthMtx[tNow,stimOn-preTime:stimOn+postTime],axis=0); 
    shape_rank = np.mean(grCe_psth[:,100:500],axis=1).argsort(); # ascending order
    shape_rank = np.flip(shape_rank); # descending order

    ### psth yMax
    tNow = np.where((ses_t['stim_cond']=='grCe') & (ses_t['target_rot']==shape_rank[0]))[0]; 
    yMax = np.max(grCe_psth);   

    ### scatter plot
    plt.figure(figsize=(7,3)); 
    plt.gcf().suptitle(f"cell#: {unitID}. Scatter plot", fontsize=14)        

    for c in np.arange(len(conditions)):
        for r in np.arange(8):
            tNow = np.where((ses_t['stim_cond']==conditions[c]) & (ses_t['target_rot']==shape_rank[r]))[0];             
            nRep = np.shape(tNow)[0];    # number of repetition

            panel_num = c*8 + r + 1; 
            plt.subplot(4,8,panel_num); 
            for t in np.arange(nRep):
                spks = np.where(spkMtx[tNow[t],200:800]==1)[0]; 
                plt.plot(spks, np.ones(np.shape(spks))*(t+1),'g.',ms=1);  # spike time stamps
            sdf_now = np.mean(psthMtx[tNow,stimOn-preTime:stimOn+postTime],axis=0); 
            plt.plot((nRep+1)*sdf_now/yMax,color=colors[c],linewidth=2);  
            plt.xlim([0,600])
            plt.ylim([0, nRep+1])
            plt.xticks([100,400],[]); 
            plt.gca().spines[['right', 'top']].set_visible(False)

            if r==0:
                plt.yticks([0,11],[0, int(yMax)]);
                if c==3:
                    plt.xticks([100,400],[0,300]);              
            else:
                plt.yticks([0,11],[]);
    plt.tight_layout()
    return 1; 

def draw_tuning_sigMod(crowd_units, crowd_trials, unitID, conditions, colors):
    stimOn = 300; 
    preTime = 100; 
    postTime = 500; 

    sesID = crowd_units.loc[unitID,'sesID']; 
    ses_t = crowd_trials[crowd_trials['sesID']==sesID].reset_index(); 
    spkMtx = crowd_units.loc[unitID,'spkMtx']; 
    psthMtx = getSDF(spkMtx,1000); 

    ### shape preference ranking
    grCe_psth = np.empty((8,600)); 
    for r in np.arange(8):
        tNow = np.where((ses_t['stim_cond']=='grCe') & (ses_t['target_rot']==r))[0]; 
        grCe_psth[r,:] = np.mean(psthMtx[tNow,stimOn-preTime:stimOn+postTime],axis=0); 
    shape_rank = np.mean(grCe_psth[:,100:500],axis=1).argsort(); # ascending order
    shape_rank = np.flip(shape_rank); # descending order

    plt.figure(figsize=(7,3)); 
    ### Tuning curves
    plt.subplot(1,2,1); 
    for c in np.arange(len(conditions)):
        resp = np.empty((8,)); 
        sem = np.empty((8,)); 
        for r in np.arange(8):
            rid = shape_rank[r]; 
            tNow = np.where((ses_t['stim_cond']==conditions[c]) & (ses_t['target_rot']==rid))[0]; 
            resp[r] = np.mean(np.mean(psthMtx[tNow,300:700],axis=1))
            sem[r] = stats.sem(np.mean(psthMtx[tNow,300:700],axis=1))
        plt.plot(resp,'o-',label=conditions[c],color=colors[c]); 
        plt.errorbar(np.arange(8),resp,sem,color=colors[c]);      
        plt.gca().spines[['right', 'top']].set_visible(False)
    plt.legend(); 

    ### significant modulation
    for c in np.arange(len(conditions)):
        pref_Mtx = np.zeros((0,630)); 
        npref_Mtx = np.zeros((0,630)); 
        for r in np.arange(8):
            rid = shape_rank[r]; 
            tNow = np.where((ses_t['stim_cond']==conditions[c]) & (ses_t['target_rot']==rid))[0]; 
            sdfNow = psthMtx[tNow,185:815]; 
            if r<4:
                pref_Mtx = np.vstack((pref_Mtx,sdfNow)); 
            else:
                npref_Mtx = np.vstack((npref_Mtx,sdfNow)); 
        # test significance
        t_signi = []; 
        for t in np.arange(600):
            tStart = t; 
            tEnd = t+30; 
            _, pval = stats.mannwhitneyu(np.mean(pref_Mtx[:,tStart:tEnd],axis=1),np.mean(npref_Mtx[:,tStart:tEnd],axis=1)); 
            if pval<0.05:
                t_signi.append(t); 
        t_signi = np.array(t_signi); 

        if c<2:
            plt.subplot(2,4,c+3); 
        else:
            plt.subplot(2,4,c+5); 
        if c==0:
            yMax = np.max(np.mean(pref_Mtx,axis=0)); 
        plt.plot(np.arange(-100,500),np.mean(pref_Mtx[:,15:615],axis=0),color=colors[c]); 
        plt.plot(np.arange(-100,500),np.mean(npref_Mtx[:,15:615],axis=0),color=colors[c],ls=':'); 
        plt.plot(t_signi-100,np.ones(np.shape(t_signi))*yMax,'k.')
        plt.ylim([0, yMax+2])
        plt.xticks([0,200,400])
        plt.gca().spines[['right', 'top']].set_visible(False)
    plt.tight_layout();         

def draw_pop_tuning(crowd_units, crowd_trials, grCe_pval, conditions, colors): 

    shape_TC = dict(); 
    shape_TC['unitID'] = []; 
    for cond_now in conditions:
        shape_TC[cond_now] = np.empty((0,8)); 
    
    for u in np.arange(len(crowd_units)):
        sesID = crowd_units.loc[u,'sesID']; 
        ses_t = crowd_trials[crowd_trials['sesID']==sesID].reset_index(); 
        
        # check two things
        # 1. all conditions were tested in this unit
        # 2. this unit is shape selective? grCe_pval < 0.05
        cond_check = 0; 
        for cond_now in conditions:
            if cond_now in ses_t['stim_cond'].values:
                cond_check += 1; 
        if (cond_check < len(conditions)) or (grCe_pval[u]>=0.05):
            continue; 

        shape_TC['unitID'].append(u); 

        spkMtx = crowd_units.loc[u,'spkMtx']; 
        psthMtx = getSDF(spkMtx,1000); 

        # shape ranking
        targ_resp = []; 
        for r in np.arange(8): 
            tNums = np.where((ses_t['stim_cond']=='grCe') & (ses_t['target_rot']==r))[0]; 
            respNow = np.mean(np.sum(crowd_units.loc[u,'spkMtx'][tNums,300:700],axis=1)*1000/400,axis=0); 
            respNow2 = 0; 
            # for color condition, shape preference is determined by gray, color target together
            if 'coCe' in conditions:
                tNums = np.where((ses_t['stim_cond']=='coCe') & (ses_t['target_rot']==r))[0]; 
                respNow2 = np.mean(np.sum(crowd_units.loc[u,'spkMtx'][tNums,300:700],axis=1)*1000/400,axis=0); 
            targ_resp.append(respNow+respNow2); 
        shape_rank = np.array(targ_resp).argsort()[::-1]; # from best to worst

        ### Tuning curves
        shape_TC_now = np.zeros((8,len(conditions))); 

        # best_grCe for normalization
        tNow = np.where((ses_t['stim_cond']=='grCe') & (ses_t['target_rot']==shape_rank[0]))[0]
        best_grCe = np.mean(np.mean(psthMtx[tNow,300:700],axis=1)); 
        
        for c in np.arange(len(conditions)):
            resp = np.empty((1,8)); 
            for r in np.arange(8):
                rid = shape_rank[r]; 
                tNow = np.where((ses_t['stim_cond']==conditions[c]) & (ses_t['target_rot']==rid))[0]; 
                try:
                    resp[0,r] = np.mean(np.mean(psthMtx[tNow,300:700],axis=1)) / best_grCe;    # normalize
                except:
                    print(tNow); 
            shape_TC[conditions[c]] = np.vstack((shape_TC[conditions[c]], resp)); 

    ### Tuning curves
    for c in np.arange(len(conditions)):
        mResp = np.mean(shape_TC[conditions[c]],axis=0); 
        sem = stats.sem(shape_TC[conditions[c]],axis=0); 
        plt.plot(mResp,'o-',label=conditions[c],color=colors[c]); 
        plt.errorbar(np.arange(8),mResp,sem,color=colors[c]);      
        plt.gca().spines[['right', 'top']].set_visible(False)
    plt.legend(); 


def draw_mod_histogram(mod_df, r_er_df, grCe_pval, stimcond, color, axis, cond2=None):
    if cond2 is None:
        chosen = np.where((grCe_pval<0.05) & (mod_df[stimcond]<0.5))[0]; 
        signi = np.where((grCe_pval<0.05) & (mod_df[stimcond]<0.5) & (r_er_df[f'{stimcond}: r2er_p']<0.05) & (r_er_df[f'{stimcond}: r_er']>0))[0]; 
    else:
        chosen = np.where((grCe_pval<0.05) & (mod_df[stimcond]<0.5) & (mod_df[cond2]<0.5))[0]; 
        signi = np.where((grCe_pval<0.05) & (mod_df[stimcond]<0.5) & (r_er_df[f'{stimcond}: r2er_p']<0.05) & (r_er_df[f'{stimcond}: r_er']>0) & (mod_df[cond2]<0.5))[0]; 

    ax1 = plt.subplot(3,4,axis); 
    counts1,bins1 = np.histogram(mod_df[stimcond].values[chosen],np.arange(-1,0.6,0.1)); 
    counts2,bins2 = np.histogram(mod_df[stimcond].values[signi],np.arange(-1,0.6,0.1)); 
    ax1.hist(mod_df[stimcond].values[chosen],np.arange(-1,0.6,0.1),facecolor=[1,1,1],edgecolor=[0,0,0],linewidth=0.5); 
    ax1.hist(mod_df[stimcond].values[signi],np.arange(-1,0.6,0.1),facecolor=color,edgecolor=[0.9,0.9,0.9],linewidth=0.5); 

    corrs = [] 
    for i in range(3):
        left = np.percentile(mod_df[stimcond].values[chosen],i*25); 
        right = np.percentile(mod_df[stimcond].values[chosen],(i)*25+50); 

        unitNow = np.where((mod_df[stimcond].values[chosen]>=left) & 
                           (mod_df[stimcond].values[chosen]<right))[0]; 
        mCorr = np.mean(r_er_df[f'{stimcond}: r_er'].values[chosen][unitNow]); 
        corrs.append([np.mean([left,right]),mCorr]); 
    corrs = np.array(corrs); 

    #ax1.plot(np.nanmedian(mod_df[stimcond].values[chosen]),np.max(counts1),'v',color=color,ms=10)
    ax1.plot(np.nanmean(mod_df[stimcond].values[chosen]),np.max(counts1),'v',color=color,ms=10)    
    ax1.set_title(stimcond)
    ax1.spines[['right', 'top']].set_visible(False)

    ax2 = ax1.twinx()  
    ax2.plot(corrs[:,0], corrs[:,1],'o-',color='C3'); 
    ax2.set_ylim([0, 1])
    ax2.tick_params(axis='y', labelcolor='C3')
    ax2.spines[['top']].set_visible(False)    


def draw_TC_shape_modulation(TC_shape_mod, grCe_pval):
    plt.figure(figsize=(8,4)); 

    # distance
    ss_units = np.where((grCe_pval<0.05))[0]
    plt.subplot(3,4,1); 
    plt.fill_between(np.arange(-100,500),100*np.sum(TC_shape_mod['grCe_SigDiff'][ss_units,:],axis=0)/len(ss_units),color=[1,0,0],alpha=0.2,linewidth=0)
    plt.plot(np.arange(-100,500),100*np.sum(TC_shape_mod['grCe_grSu18Fa_SigDiff'][ss_units,:],axis=0)/len(ss_units),color=[0.8,0.8,0.8])
    plt.xticks(np.arange(0,500,200))
    plt.title('grCe_grSu18Fa')
    plt.gca().spines[['right', 'top']].set_visible(False)
    plt.subplot(3,4,5); 
    plt.fill_between(np.arange(-100,500),100*np.sum(TC_shape_mod['grCe_SigDiff'][ss_units,:],axis=0)/len(ss_units),color=[1,0,0],alpha=0.2,linewidth=0)
    plt.plot(np.arange(-100,500),100*np.sum(TC_shape_mod['grCe_grSu12Mi_SigDiff'][ss_units,:],axis=0)/len(ss_units),color=[0.4,0.4,0.4])
    plt.xticks(np.arange(0,500,200))
    plt.title('grCe_grSu12Mi')
    plt.gca().spines[['right', 'top']].set_visible(False)
    plt.subplot(3,4,9); 
    plt.fill_between(np.arange(-100,500),100*np.sum(TC_shape_mod['grCe_SigDiff'][ss_units,:],axis=0)/len(ss_units),color=[1,0,0],alpha=0.2,linewidth=0)
    plt.plot(np.arange(-100,500),100*np.sum(TC_shape_mod['grCe_grSu6Ne_SigDiff'][ss_units,:],axis=0)/len(ss_units),color=[0,0,0])
    plt.xticks(np.arange(0,500,200))
    plt.title('grCe_grSu6Ne')
    plt.gca().spines[['right', 'top']].set_visible(False)

    # number
    ss_units = np.where((grCe_pval<0.05))[0]
    plt.subplot(3,4,2); 
    plt.fill_between(np.arange(-100,500),100*np.sum(TC_shape_mod['grCe_SigDiff'][ss_units,:],axis=0)/len(ss_units),color=[1,0,0],alpha=0.2,linewidth=0)
    plt.plot(np.arange(-100,500),100*np.sum(TC_shape_mod['grCe_grSu1Ne_SigDiff'][ss_units,:],axis=0)/len(ss_units),color=[0.8,0.8,0.8])
    plt.xticks(np.arange(0,500,200))
    plt.title('grCe_grSu1Ne')
    plt.gca().spines[['right', 'top']].set_visible(False)
    plt.subplot(3,4,6); 
    plt.fill_between(np.arange(-100,500),100*np.sum(TC_shape_mod['grCe_SigDiff'][ss_units,:],axis=0)/len(ss_units),color=[1,0,0],alpha=0.2,linewidth=0)
    plt.plot(np.arange(-100,500),100*np.sum(TC_shape_mod['grCe_grSu3Ne_SigDiff'][ss_units,:],axis=0)/len(ss_units),color=[0.4,0.4,0.4])
    plt.xticks(np.arange(0,500,200))
    plt.title('grCe_grSu3Ne')
    plt.gca().spines[['right', 'top']].set_visible(False)
    plt.subplot(3,4,10); 
    plt.fill_between(np.arange(-100,500),100*np.sum(TC_shape_mod['grCe_SigDiff'][ss_units,:],axis=0)/len(ss_units),color=[1,0,0],alpha=0.2,linewidth=0)
    plt.plot(np.arange(-100,500),100*np.sum(TC_shape_mod['grCe_grSu6Ne_SigDiff'][ss_units,:],axis=0)/len(ss_units),color=[0,0,0])
    plt.xticks(np.arange(0,500,200))
    plt.title('grCe_grSu6Ne')
    plt.gca().spines[['right', 'top']].set_visible(False)

    # color
    ss_units = np.where((grCe_pval<0.05) & (np.isnan(np.sum(TC_shape_mod['coCe_SigDiff'],axis=1))==0))[0]
    plt.subplot(3,4,3); 
    plt.fill_between(np.arange(-100,500),100*np.sum(TC_shape_mod['grCe_SigDiff'][ss_units,:],axis=0)/len(ss_units),color=[1,0,0],alpha=0.2,linewidth=0)
    plt.plot(np.arange(-100,500),100*np.sum(TC_shape_mod['grCe_grSu6Ne_SigDiff'][ss_units,:],axis=0)/len(ss_units),color=[0,0,0])
    plt.xticks(np.arange(0,500,200))
    plt.title('grCe_grSu6Ne')
    plt.gca().spines[['right', 'top']].set_visible(False)
    plt.subplot(3,4,7); 
    plt.fill_between(np.arange(-100,500),100*np.sum(TC_shape_mod['grCe_SigDiff'][ss_units,:],axis=0)/len(ss_units),color=[1,0,0],alpha=0.2,linewidth=0)
    plt.plot(np.arange(-100,500),100*np.sum(TC_shape_mod['coCe_SigDiff'][ss_units,:],axis=0)/len(ss_units),color=[0,0,1])
    plt.xticks(np.arange(0,500,200))
    plt.title('coCe')
    plt.gca().spines[['right', 'top']].set_visible(False)
    plt.subplot(3,4,11); 
    plt.fill_between(np.arange(-100,500),100*np.sum(TC_shape_mod['grCe_SigDiff'][ss_units,:],axis=0)/len(ss_units),color=[1,0,0],alpha=0.2,linewidth=0)
    plt.plot(np.arange(-100,500),100*np.sum(TC_shape_mod['coCe_grSu6Ne_SigDiff'][ss_units,:],axis=0)/len(ss_units),color=[0.5,0.5,1])
    plt.xticks(np.arange(0,500,200))
    plt.title('coCe_grSu6Ne')
    plt.gca().spines[['right', 'top']].set_visible(False)

    # shape
    ss_units = np.where(
        (grCe_pval<0.05) & (np.isnan(np.sum(TC_shape_mod['grCe_grSu12SmCiNe_SigDiff'],axis=1))==0)
        & (np.isnan(np.sum(TC_shape_mod['grCe_grSu6CiNe_SigDiff'],axis=1))==0))[0]
    plt.subplot(3,4,4); 
    plt.fill_between(np.arange(-100,500),100*np.sum(TC_shape_mod['grCe_SigDiff'][ss_units,:],axis=0)/len(ss_units),color=[1,0,0],alpha=0.2,linewidth=0)
    plt.plot(np.arange(-100,500),100*np.sum(TC_shape_mod['grCe_grSu12SmCiNe_SigDiff'][ss_units,:],axis=0)/len(ss_units),color=[0.8,0.8,0.8])
    plt.xticks(np.arange(0,500,200))
    plt.title('grCe_grSu12SmCiNe')
    plt.gca().spines[['right', 'top']].set_visible(False)
    plt.subplot(3,4,8); 
    plt.fill_between(np.arange(-100,500),100*np.sum(TC_shape_mod['grCe_SigDiff'][ss_units,:],axis=0)/len(ss_units),color=[1,0,0],alpha=0.2,linewidth=0)
    plt.plot(np.arange(-100,500),100*np.sum(TC_shape_mod['grCe_grSu6CiNe_SigDiff'][ss_units,:],axis=0)/len(ss_units),color=[0.4,0.4,0.4])
    plt.xticks(np.arange(0,500,200))
    plt.title('grCe_grSu6CiNe')
    plt.gca().spines[['right', 'top']].set_visible(False)
    plt.subplot(3,4,12); 
    plt.fill_between(np.arange(-100,500),100*np.sum(TC_shape_mod['grCe_SigDiff'][ss_units,:],axis=0)/len(ss_units),color=[1,0,0],alpha=0.2,linewidth=0)
    plt.plot(np.arange(-100,500),100*np.sum(TC_shape_mod['grCe_grSu6Ne_SigDiff'][ss_units,:],axis=0)/len(ss_units),color=[0,0,0])
    plt.xticks(np.arange(0,500,200))
    plt.title('grCe_grSu6Ne')
    plt.gca().spines[['right', 'top']].set_visible(False)

    plt.tight_layout()

#%% two dimensional gaussian function with rotation    
def twoD_Gaussian(posData, amplitude, x0, y0, sigma_x, sigma_y, theta, offset):

    """
    https://en.wikipedia.org/wiki/Gaussian_function

    a =  cos(theta)**2 / (2 * sigma_x**2) + 
        sin(theta)**2 / (2 * sigma_y**2)

    b = - (- sin(2*theta) / (4 * sigma_x**2) 
           + sin(2*theta) / (4 * sigma_y**2))

    c = sin(theta)**2 / (2 * sigma_x**2) + 
        cos(theta)**2 / (2 * sigma_y**2)

    f(x,y) = A * exp ( 
                      -( a * (x - x0)**2 
                      + 2 * b * (x - x0) * (y - y0) 
                      + c * (y - y0)**2 )
                      ) 
             + offset

        theta: counter clockwise rotation angle 
               (for counter clockwise, invert sign of 'b')
        sigma_x, sigma_y: spreads of the blob

        A: amplitude
        x0, y0 is the center
    """

    (x, y) = posData; 
    x0 = float(x0); 
    y0 = float(y0);         
    
    a = ( (np.cos(theta)**2 / (2*(sigma_x**2))) 
           + (np.sin(theta)**2 / (2*sigma_y**2)) ); 

    b = - ( - (np.sin(2*theta) / (4*sigma_x**2)) 
            + (np.sin(2*theta) / (4*sigma_y**2)) ); 

    c = ( (np.sin(theta)**2 / (2*sigma_x**2)) 
          + (np.cos(theta)**2 / (2*sigma_y**2)) ); 

    f_xy = ( amplitude * np.exp( -((a * (x-x0)**2) 
                                   + (2*b * (x-x0) * (y-y0)) 
                                   + (c * (y-y0)**2)) )
             + offset ) 

    # constraint to make sigma_x > sigma_y 
    # this is critically associated with theta
    if sigma_y > sigma_x: 
        penalty = (sigma_y - sigma_x) * 100; 
    else:
        penalty = 0; 

    #f_xy = f_xy + penalty; 

    # Return a contiguous flattened array.
    return f_xy.ravel(); 


def draw_RFmap(rf_data, unitID):

    xRange = rf_data['xRange'][unitID]
    yRange = rf_data['yRange'][unitID]
\
    ## position for drawing
    x2 = np.arange(xRange[0],xRange[1]+0.1,0.1); 
    y2 = np.arange(yRange[0],yRange[1]+0.1,0.1); 
    x2, y2 = np.meshgrid(x2,y2); 
    posData2 = np.vstack((x2.ravel(),y2.ravel())); 
    RFmap_norm = rf_data['RFmap_norm'][unitID].reshape((yRange[1]-yRange[0]+1,xRange[1]-xRange[0]+1)); 
    deg_now = rf_data['p_theta'][unitID]*180/np.pi;               
    cmass = rf_data['maxPos'][unitID]; 
    fit_r = rf_data['fit_r'][unitID]; 

    # parameters: amplitude, x0, y0, sigma_x, sigma_y, theta, offset
    popt = [
        rf_data['p_amp'][unitID],
        rf_data['p_x0'][unitID],
        rf_data['p_y0'][unitID],        
        rf_data['p_sigx'][unitID],
        rf_data['p_sigy'][unitID],        
        rf_data['p_theta'][unitID],
        rf_data['p_offset'][unitID],        
    ]
    
    plt.figure(figsize=(5,2.5)); 
    plt.subplot(1,2,1); 
    plt.imshow(RFmap_norm, origin='lower'); 
    plt.xticks(np.arange(7),np.arange(xRange[0],xRange[1]+1,1)); 
    plt.yticks(np.arange(7),np.arange(yRange[0],yRange[1]+1,1)); 
    plt.title(rf_data['sesName'][unitID])

    plt.subplot(1,2,2); 
    data_fitted = twoD_Gaussian(posData2, *popt); 
    ## will use 2.35 SD as a measure of the RF diameter (FWHH)
    rf_level = twoD_Gaussian([popt[1]+1.17*popt[3]*np.cos(popt[5]),
                                popt[2]+1.17*popt[3]*np.sin(popt[5])], *popt)      
    RFmap_spline  = rf_data['RFmap_spline'][unitID].reshape(
        int(len(posData2[0])**(1/2)),
        int(len(posData2[0])**(1/2))
    )
    #plt.imshow(RFmap_spline,
    #           origin='lower',extent=(xRange[0],xRange[1],yRange[0],yRange[1]));         
    plt.imshow(data_fitted.reshape(int(len(posData2[0])**(1/2)),
                                   int(len(posData2[0])**(1/2))),
               origin='lower',extent=(xRange[0],xRange[1],yRange[0],yRange[1]));         
    plt.contour(x2, y2, data_fitted.reshape(int(len(posData2[0])**(1/2)),
                                            int(len(posData2[0])**(1/2))), 
                [rf_level], colors='w')
    plt.plot(popt[1],popt[2],'ro')            
    plt.plot(popt[1] + 1.17*popt[3]*np.cos(popt[5]),
                popt[2] + 1.17*popt[3]*np.sin(popt[5]),'m*')                                            
    plt.plot(popt[1] - 1.17*popt[4]*np.sin(popt[5]),
                popt[2] + 1.17*popt[4]*np.cos(popt[5]),'m*')       
    plt.plot(cmass[0],cmass[1],'bo')
    plt.title(f'theta = {deg_now:.2f} deg'); 
    plt.xlim(xRange[0]-0.5,xRange[1]+0.5); 
    plt.ylim(yRange[0]-0.5,yRange[1]+0.5);    
    plt.tight_layout(); 
