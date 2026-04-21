import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
from astropy import time, coordinates as co, units as u
import os

def weighted_mean(values, weights,axis=None):
    return np.nansum(values * weights,axis) / np.nansum(weights,axis)


class Ensemble():
    def __init__(self, config=None, save_path = '.', diagnostics = False):
        self.save_path = save_path
        self.diagnostics = diagnostics
        self.open_df = pd.DataFrame()
        self.lines = []
        
        # self.location = 'TNO' #None    #site name or 'Alt Az Elv'
        self.location = co.EarthLocation.of_site('TNO')
        # self.src_pos = 
        self.src_pos = co.SkyCoord('06 45 08.92 -16 42 58.0', unit=(u.hourangle, u.deg))
        self.exc_frame = None
        self.sigma_clip = 3
        self.diagnostics=diagnostics
        
        if config:
            self.read_config_file(config)
        
    def read_config_file(self, config):
        try:
            with open(config, 'r') as conf:
                self.lines = conf.readlines()
            
            for line in self.lines:
                if len(line.split()) == 0:
                    continue
                    
            par = line.split()[0]

            if par.lower() == 'location' or par.lower() == 'loc':
                try:
                    self.location = co.EarthLocation.of_site(line.split()[1])
                except co.errors.UnknownSiteException:
                    try:
                        self.location = co.EarthLocation.from_geodetic(line.split()[1], line.split()[2], height=line.split()[3])
                    except:
                        warn('Location {0:s} not recognised, '
                             'proceeding without barycentering!'\
                             .format(line.split()[1]))
                        
            elif par.lower() == 'exc_frame':
                if self.exc_frame is None:
                    self.exc_frame = {}
                self.exc_frame[line.split()[1]] = line.split()[2:]
                
            elif par.lower() == 'sigma_clip':
                self.sigma_clip = float(line.split()[1])

            elif par.lower() == 'pos':
                self.src_pos = co.SkyCoord(' '.join(line.split()[1:]),
                                           unit=(u.hourangle,u.deg))



                
        except FileNotFoundError:
            print(f"Error: Config file {config} not found.")


    
    def read_log_file(self, logfile, instrument="hcam"):
        headers = None
        
        try:
            with open(logfile, 'r') as logf:
                for line in logf:
                    line_strip = line.strip()
                    if line_strip.startswith("#") and " = CCD" in line_strip:

                        parts = line_strip.split()
                        headers = parts[3:] 
                        break 
        except FileNotFoundError:
            print(f"Error: Log file {logfile} not found.")
            return

        if headers:
            target_prefixes = ('counts_', 'countse_', 'sky_', 'flag_')
            base_cols = ['CCD', 'nframe', 'MJD', 'MJDok', 'Exptim']
            
            cols_to_keep = [c for c in headers if c in base_cols or c.startswith(target_prefixes)]
            
            self.open_df = pd.read_csv(
                logfile, 
                sep=r'\s+', 
                comment='#', 
                names=headers, 
                header=None, 
                usecols=cols_to_keep, 
                float_precision='high'
            )
            # remove MJDok == 0
            self.open_df = self.open_df[self.open_df['MJDok'] != 0].copy()
            
            # correct time MJD to BJD
            self.barycenter_times()
            
            # remove flag_* != 0
            self.filter_data()
            self.get_airmass()
            
        else:
            print("Could not find the header line (starting with '#' and containing ' = CCD')")

    def filter_data(self):
        if self.open_df.empty:
            print("No data to filter.")
            return

        flag_cols = [c for c in self.open_df.columns if c.startswith('flag_')]
        
        for f_col in flag_cols:
            star_idx = f_col.split('_')[1]
            
            counts_col = f'counts_{star_idx}'
            countse_col = f'countse_{star_idx}'
            sky_col = f'sky_{star_idx}'
            
            bad_data_mask = self.open_df[f_col] != 0
            
            for col in [counts_col, countse_col, sky_col]:
                if col in self.open_df.columns:
                    self.open_df.loc[bad_data_mask, col] = np.nan
                
        # print(f"Filtering complete: Bad data (flag != 0) replaced with NaN.")

    def get_instrumental_mags(self, data, diagnostics=False):
        nstars = len([c for c in self.open_df.columns if c.startswith('counts_')])
        new_columns = {}
        for n in range(1, nstars + 1):
            counts = data[f'counts_{n}']
            countse = data[f'countse_{n}']
            exptim = data['Exptim']
            
            safe_counts = np.where(counts > 0, counts, np.nan)
            
            new_columns[f'instrumag_{n}'] = -2.5 * np.log10(safe_counts / exptim)
            new_columns[f'einstrumag_{n}'] = (2.5 / np.log(10)) * (countse / safe_counts)
            
        new_df = pd.DataFrame(new_columns, index=data.index)
        data = pd.concat([data, new_df], axis=1)
        
        return data

    def barycenter_times(self,):
        # self.open_df['MJD']
        t = time.Time(self.open_df['MJD'], scale='utc',format='mjd', location=self.location)
        ssbcorr = t.light_travel_time(self.src_pos)
        self.open_df['BJD'] = (t.tdb + ssbcorr).value
        

    def fit_airmass_coeff1(self, ee, weights, airmass):
        exp_weight = np.nansum(weights, axis=1)
        #    [ [sum(w), sum(w*x)], [sum(w*x), sum(w*x^2)] ]
        M = np.zeros((2, 2))
        M[0, 0] = np.nansum(exp_weight)
        M[0, 1] = np.nansum(exp_weight * airmass)
        M[1, 0] = M[0, 1] 
        M[1, 1] = np.nansum(exp_weight * (airmass ** 2.0))
        R = np.array([np.nansum(exp_weight * ee),
                      np.nansum(exp_weight * ee * airmass)])
        P = np.linalg.solve(M, R)   #P = [Intercept, Slope]
        
        return P
        
    def solve_ensemble(self, data, ):
        mag_cols = [c for c in data.columns if c.startswith('instrumag_')]
        err_cols = [c for c in data.columns if c.startswith('einstrumag_')]
        # err_cols = [c.replace('instrumag_', 'einstrumag_') for c in mag_cols]
        

        m_obs = data[mag_cols].values
        err_obs = data[err_cols].values

        W_all = np.where(np.isnan(err_obs) | (err_obs == 0), 0.0, 1.0 / (err_obs**2))
        
        mean_star_1 = weighted_mean(m_obs[:, 0], W_all[:, 0])
        if np.isnan(mean_star_1): mean_star_1 = 0.0

        m_obs -= mean_star_1
        
        M_obs_all = np.nan_to_num(m_obs, nan=0.0)
        x=True
        if x:
            E, S = M_obs_all.shape
            A = S - 1  
            num_unknowns = A + E
            W_solve = W_all[:, 1:] 
            M_solve = M_obs_all[:, 1:]
    
        # elif:
        #     E, S = M_obs_all.shape
        #     A = S 
        #     num_unknowns = A + E
        #     W_solve = W_all[:, :] 
        #     M_solve = M_obs_all[:, :]

        M = np.zeros((num_unknowns, num_unknowns))
        R = np.zeros(num_unknowns)


        np.fill_diagonal(M[:A, :A], np.sum(W_solve, axis=0))
        np.fill_diagonal(M[A:, A:], np.sum(W_all, axis=1))

        M[:A, A:] = W_solve.T
        M[A:, :A] = W_solve

        R[:A] = np.sum(M_solve * W_solve, axis=0)
        R[A:] = np.sum(M_obs_all * W_all, axis=1)


        # theta = np.linalg.solve(M, R)
        theta, _, _, _ = np.linalg.lstsq(M, R, rcond=None)
        meanmags = np.zeros(S)
        meanmags[0] = mean_star_1
        meanmags[1:] = theta[:A] + mean_star_1


        #Extinction by ATM
        data['exposure_corr'] = theta[A:]
        ee_total = data['exposure_corr'].values

        # Find kappa (Extinction Coefficient)
        self.offset, self.kappa = self.fit_airmass_coeff1(data['exposure_corr'].values, W_all, data['secz'].values)

        ee_cloud = ee_total - (self.kappa * data['secz'].values + self.offset)
        meanmags += self.offset
        
        m_obs_orig = m_obs + mean_star_1

        resids = (m_obs_orig 
                  - meanmags[np.newaxis, :] 
                  - (self.kappa * data['secz'].values)[:, np.newaxis] 
                  - ee_cloud[:, np.newaxis])
        return resids, data,

        
    def get_airmass(self):
        if self.open_df.empty:
            return
        times = time.Time(self.open_df['MJD'], format='mjd', scale='utc', location=self.location)
    
        altaz_frame = co.AltAz(obstime=times, location=self.location)
        target_altaz = self.src_pos.transform_to(altaz_frame)
        self.open_df['secz'] = target_altaz.secz.value
        
        return self.open_df

    
    def find_most_variable_star(self, df):
        mag_cols = [c for c in df.columns if c.startswith('instrumag_')]
        if not mag_cols: return {}


        norm_df = df[mag_cols].apply(lambda x: x - np.nanmedian(x), axis=0)
        
        variability_results = {}
        show_diag = getattr(self, 'diagnostics', False)


        if show_diag:
            n_stars = len(mag_cols)
            ncols = 4
            nrows = int(np.ceil(n_stars / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(20, nrows * 3.5), sharex=True)
            if n_stars == 1: axes = [axes] # Handle case with only 1 star
            axes = axes.flatten()
            print(f"📊 Analyzing & Plotting {n_stars} stars for variability...")


        for i, target_col in enumerate(mag_cols):
            star_id = target_col.split('_')[-1]
            
            other_stars = [c for c in mag_cols if c != target_col]
            if other_stars:
                master_ref = norm_df[other_stars].mean(axis=1)
                residual = norm_df[target_col] - master_ref
            else:

                residual = norm_df[target_col] - norm_df[target_col]
            

            sd_val = np.nanstd(residual)
            variability_results[target_col] = sd_val


            if show_diag:
                ax = axes[i]
                ax.plot(df.index, residual, color='indigo', lw=0.8, alpha=0.7)
                ax.axhline(0, color='red', linestyle='--', alpha=0.3)
                
                ax.set_title(f"Star {star_id}\nResid SD: {sd_val:.4f}", 
                             fontsize=11, fontweight='bold', color='darkblue')
                
                ax.invert_yaxis() # Magnitude scale
                ax.grid(True, linestyle=':', alpha=0.5)

                if i % ncols == 0:
                    ax.set_ylabel('Diff Mag', fontsize=9)


        if show_diag:
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])

            fig.suptitle('Ensemble Residual Diagnostics (Target Star vs. Others Mean)', 
                         fontsize=22, fontweight='bold', y=1.02)
            plt.tight_layout()
            plt.show()


        return dict(sorted(variability_results.items(), key=lambda item: item[1], reverse=True))
        
    
    def filter_by_sky(self, df, tolerance=0.03):
        import matplotlib.pyplot as plt
        import numpy as np

        lower_limit = 1.0 - tolerance
        upper_limit = 1.0 + tolerance
        
        sky_cols = [c for c in df.columns if c.startswith('sky_')]
        if not sky_cols:
            return []

        median_sky_rate = df[sky_cols].median(axis=1) / df['Exptim']
        stars_to_drop = []
        star_diagnostics_data = {}

        for col in sky_cols:
            star_num = col.split('_')[1]
            star_sky_rate = df[col] / df['Exptim']
            ratio_series = star_sky_rate / median_sky_rate
            rel_sky_med = ratio_series.median()
            
            star_diagnostics_data[star_num] = {
                'series': ratio_series,
                'median': rel_sky_med
            }

           
            if rel_sky_med < lower_limit or rel_sky_med > upper_limit:
                stars_to_drop.append(star_num)

        if getattr(self, 'diagnostics', False):
            n_stars = len(sky_cols)
            ncols = 4
            nrows = int(np.ceil(n_stars / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(20, nrows * 3.5), sharex=True)
            axes = axes.flatten()

            for i, (star_num, info) in enumerate(star_diagnostics_data.items()):
                ax = axes[i]
                rel_sky = info['median']
                
                ax.plot(df.index, info['series'], color='seagreen', lw=0.8, alpha=0.5)
                
                ax.axhline(upper_limit, color='crimson', ls='-', lw=1, alpha=0.5, label=f'+{tolerance}')
                ax.axhline(lower_limit, color='crimson', ls='-', lw=1, alpha=0.5, label=f'-{tolerance}')
                ax.axhline(1.0, color='black', ls=':', alpha=0.8)

                ax.axhline(rel_sky, color='darkorange', ls='--', lw=1.5)

                ax.set_title(f"Star {star_num}\nRatio: {rel_sky:.3f}", fontsize=10, fontweight='bold')

                if rel_sky < lower_limit or rel_sky > upper_limit:
                    ax.set_facecolor('#fff0f0') 
                    ax.title.set_color('crimson')

            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])

            fig.suptitle(f'Symmetric Sky Diagnostics ($1.0 \\pm {tolerance}$)', fontsize=22, fontweight='bold', y=1.02)
            plt.tight_layout()
            plt.show()

        return stars_to_drop

        
    def filter_by_sky(self, df, tolerance=0.03):

        lower_limit = 1.0 - tolerance
        upper_limit = 1.0 + tolerance
        sky_cols = [c for c in df.columns if c.startswith('sky_')]
        if not sky_cols: return []

        median_sky_rate = df[sky_cols].median(axis=1) / df['Exptim']
        stars_to_drop_entirely = []
        star_diagnostics_data = {}

        for col in sky_cols:
            star_num = col.split('_')[1]
            mag_col, err_col = f'instrumag_{star_num}', f'einstrumag_{star_num}'
            
            star_sky_rate = df[col] / df['Exptim']
            ratio_series = star_sky_rate / median_sky_rate
            
            # Point-by-point cleaning
            bad_mask = (ratio_series < lower_limit) | (ratio_series > upper_limit)
            if bad_mask.any():
                df.loc[bad_mask, mag_col] = np.nan
                if err_col in df.columns:
                    df.loc[bad_mask, err_col] = np.nan

            rel_sky_med = ratio_series.median()
            star_diagnostics_data[star_num] = {'series': ratio_series, 'median': rel_sky_med, 'bad_mask': bad_mask}
            
            if rel_sky_med < lower_limit or rel_sky_med > upper_limit:
                stars_to_drop_entirely.append(star_num)



        if getattr(self, 'diagnostics', False):
            n_stars = len(sky_cols)
            ncols = 4
            nrows = int(np.ceil(n_stars / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(20, nrows * 3.5), sharex=True)
            axes = axes.flatten()
            
            for i, (star_num, info) in enumerate(star_diagnostics_data.items()):
                ax = axes[i]
                
                # --- Check if this star is flagged for removal ---
                is_removed = star_num in stars_to_drop_entirely
                
                line_color = 'crimson' if is_removed else 'seagreen'
                bg_color   = '#fff0f0' if is_removed else 'white'
                
                ax.set_facecolor(bg_color)
                ax.plot(df.index, info['series'], color=line_color, lw=0.8, alpha=0.6)
                
                # bad points (red dots)
                if info['bad_mask'].any():
                    ax.scatter(df.index[info['bad_mask']], 
                              info['series'][info['bad_mask']], 
                              color='red', s=10, zorder=5)
                
                ax.axhline(upper_limit, color='crimson', ls='-', lw=1, alpha=0.3)
                ax.axhline(lower_limit, color='crimson', ls='-', lw=1, alpha=0.3)
                
                # --- Title: red + REMOVED tag if flagged ---
                title_text  = f"Star {star_num}\nRatio: {info['median']:.3f}"
                title_color = 'crimson' if is_removed else 'black'
                
                if is_removed:
                    title_text += "\n REMOVED"
                
                ax.set_title(title_text, fontsize=10, fontweight='bold', color=title_color)
            
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])
            
            fig.suptitle(rf'Sky Point-by-Point Cleaning ($\pm {tolerance}$)', fontsize=22, y=1.02)
            plt.tight_layout()
            plt.show()
        
        return stars_to_drop_entirely

        
    def filter_by_contrast(self, df, min_contrast=5.0):
        import matplotlib.pyplot as plt
        import numpy as np

        counts_cols = [c for c in df.columns if c.startswith('counts_')]
        stars_to_drop = []
        contrast_data = {}

        for col in counts_cols:
            star_num = col.split('_')[1]
            sky_col = f'sky_{star_num}'
            
            if sky_col in df.columns:
                contrast_series = df[col] / (df[sky_col] + 1e-10)
                med_contrast = contrast_series.median()
                
                contrast_data[star_num] = med_contrast
                
                if med_contrast < min_contrast:
                    stars_to_drop.append(star_num)

        if getattr(self, 'diagnostics', False):
            plt.figure(figsize=(10, 5))
            nums = list(contrast_data.keys())
            vals = list(contrast_data.values())
            colors = ['crimson' if v < min_contrast else 'seagreen' for v in vals]
            
            plt.bar(nums, vals, color=colors, alpha=0.7)
            plt.axhline(min_contrast, color='red', ls='--', label=f'Min Threshold ({min_contrast}x)')
            plt.yscale('log') # ใช้ Log scale เพราะ Contrast ดาวสว่างกับดาวจางต่างกันมาก
            plt.title('Star Contrast Ratio (Counts / Sky)')
            plt.ylabel('Contrast Ratio (Log Scale)')
            plt.xlabel('Star ID')
            plt.legend()
            plt.show()

        return stars_to_drop

    
    def plot_comparison_lr(self, star_id):
        raw_col = f'instrumag_{star_id}'
        real_col = f'realmag_{star_id}'
        
       
        if raw_col not in self.df_keep.columns:
            print(f"not found {star_id}")
            return

        # --- LEFT PLOT: (Relative Mag) ---
        raw_data = self.df_keep[raw_col].dropna()
        raw_norm = raw_data - raw_data.median() 
        raw_rms = np.nanstd(raw_norm)


        if hasattr(self, 'real_mag_data') and real_col in self.real_mag_data.columns:
            has_cal = True
            cal_data = self.real_mag_data[real_col].dropna()
            
         
            cal_rms = np.nanstd(cal_data) 
            y_label_right = 'Apparent Magnitude (mag)'
            
        elif 'exposure_corr' in self.data.columns:

            has_cal = True
            cal_data = self.df_keep[raw_col] - self.data['exposure_corr']
            cal_data = cal_data.dropna()
            cal_data = cal_data - cal_data.median()
            cal_rms = np.nanstd(cal_data)
            y_label_right = 'Relative Magnitude (Calibrated)'
            print(" กราฟขวาแสดงผลแบบ Relative (ยังไม่ได้ใส่ Zero Point offset)")
        else:
            has_cal = False
            print(" ยังไม่ได้รัน solve_ensemble")
        

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6)) # ห้ามใส่ sharey=True เพราะแกน Y ไม่เท่ากันแล้ว

        ax1.scatter(raw_data.index, raw_norm, s=10, color='gray', alpha=0.5, label='Raw Data')
        ax1.set_title(f'BEFORE: Raw Star {star_id}\nRMS = {raw_rms:.4f}', fontsize=14, fontweight='bold', color='darkred')
        ax1.set_ylabel('Relative Magnitude (Raw)', fontsize=12)
        ax1.set_xlabel('Frame Number', fontsize=12)
        ax1.invert_yaxis()
        ax1.grid(True, ls=':', alpha=0.6)

        #  (Calibrated + Offset)
        if has_cal:
            ax2.scatter(cal_data.index, cal_data, s=10, color='royalblue', alpha=0.8, label='Ensemble + ZP')
            ax2.set_title(f'AFTER: Apparent Mag Star {star_id}\nRMS = {cal_rms:.4f}', fontsize=14, fontweight='bold', color='darkgreen')
            ax2.set_ylabel(y_label_right, fontsize=12)
            
            improvement = (1 - (cal_rms / raw_rms)) * 100
            ax2.text(0.05, 0.95, f'Improvement: {improvement:.1f}%', transform=ax2.transAxes, 
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax2.text(0.5, 0.5, 'NO DATA', ha='center', va='center', fontsize=16, color='red')

        ax2.set_xlabel('Frame Number', fontsize=12)
        if has_cal: ax2.invert_yaxis()
        ax2.grid(True, ls=':', alpha=0.6)

        fig.suptitle(f'Photometry Results: Star {star_id}', fontsize=18, y=1.05)
        plt.tight_layout()
        plt.show()
        

    def get_real_mags_for_all(self, ref_star_id, catalog_mag):
    
        inst_col = f'instrumag_{ref_star_id}'
        if inst_col not in self.data.columns:
            print(f"star {ref_star_id} was Filter")
            return None
            

        corrected_ref_mag = self.data[inst_col] - self.data['exposure_corr']
        zp = catalog_mag - corrected_ref_mag.median()
        print(f" Zero Point calculated from Star {ref_star_id}: {zp:.4f}")


        mag_cols = [c for c in self.df_keep.columns if c.startswith('instrumag_')]
        real_mag_df = pd.DataFrame(index=self.df_keep.index)
        
        if 'MJD' in self.df_keep.columns:
            real_mag_df['MJD'] = self.df_keep['MJD']
        elif 'BJD' in self.df_keep.columns:
             real_mag_df['BJD'] = self.df_keep['BJD']
        else:
             real_mag_df['time'] = self.df_keep.index

        for col in mag_cols:
            star_id = col.split('_')[-1]
            real_mag_df[f'realmag_{star_id}'] = self.df_keep[col] - self.data['exposure_corr'] + zp
            


        self.real_mag_data = real_mag_df
        return real_mag_df



    def plot_all_comparison_lr(self, save_folder=None, all_stars = True, xlim=None):
        if all_stars :
            mag_cols = [c for c in self.df_keep.columns if c.startswith('instrumag_')]
        else:
            mag_cols = self.surviving_cols
            
        if not mag_cols:
            print(" ")
            return

        has_cal = 'exposure_corr' in self.data.columns
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 15)) 

        raw_rms_list = []
        cal_rms_list = []
        colors = plt.cm.turbo(np.linspace(0, 1, len(mag_cols)))
        
        for i, col in enumerate(mag_cols):
            star_id = col.split('_')[-1] 
            star_color = colors[i]
            err_col = f'einstrumag_{star_id}'
            valid_idx = self.df_keep[col].dropna().index
            # --- (RAW) ---
            raw_data = self.df_keep.loc[valid_idx, col]
            raw_err = self.df_keep.loc[valid_idx, err_col]
            
            if len(raw_data) > 0:
                raw_rms_list.append(np.nanstd(raw_data))
                ax1.errorbar(raw_data.index, raw_data, yerr=raw_err, fmt='o', 
                             markersize=2, color=star_color, ecolor=star_color, 
                             alpha=0.5, elinewidth=0.8, capsize=0, zorder=2)
                # ax1.scatter(raw_data.index, raw_data, s=4, color=star_color, alpha=0.7)
                # Add mean/zero reference line for RAW
                ax1.axhline(np.median(raw_data), color=star_color, linestyle='--', alpha=0.3, zorder=1)

            # --- (CALIBRATED) ---
            if has_cal:
                cal_data = self.df_keep[col] - self.data['exposure_corr']
                
                # Get valid indices so mag and error arrays perfectly match
                valid_cal_idx = cal_data.dropna().index
                cal_data = cal_data.loc[valid_cal_idx]
                cal_err = self.df_keep.loc[valid_cal_idx, err_col] 
                
                if len(cal_data) > 0:
                    cal_rms_list.append(np.nanstd(cal_data))
                    ax2.errorbar(cal_data.index, cal_data, yerr=cal_err, fmt='o-', 
                                 markersize=3, color=star_color, ecolor=star_color, 
                                 alpha=0.6, elinewidth=0.8, capsize=0, 
                                 label=f'Star {star_id}', zorder=2)
                    
                    ax2.axhline(np.median(cal_data), color=star_color, linestyle='--', alpha=0.3, zorder=1)
            
            
        ax1.set_title('BEFORE: Raw Instrumental Mag', fontsize=14, fontweight='bold')
        ax1.invert_yaxis()
        ax1.grid(True, ls=':', alpha=0.6)
        if xlim is not None:
            ax1.set_xlim(xlim)
            
        if has_cal:
            ax2.set_title('AFTER: Ensemble Calibrated Mag', fontsize=14, fontweight='bold')
            ax2.invert_yaxis()
            ax2.grid(True, ls=':', alpha=0.6)
            

            ax2.legend(loc='upper left', bbox_to_anchor=(1.02, 1), 
                       ncol=2, fontsize=8, markerscale=2, title="Star IDs")
            if xlim is not None:
                ax2.set_xlim(xlim)

        if has_cal:
            # Plot the frame-by-frame mean ensemble correction
                ax3.plot(self.data.index, self.data['exposure_corr'], 
                         color='teal', marker='.', markersize=4, linestyle='-', alpha=0.7, 
                         label='Exposure Correction (Mean Level)')
                
                ax3.set_title('Atmospheric Correction (Mean Level)', fontsize=14, fontweight='bold')
                ax3.set_ylabel('Correction (mag)', fontsize=12)
                ax3.set_xlabel('Frame Number', fontsize=12)
                ax3.invert_yaxis()
                ax3.grid(True, ls=':', alpha=0.6)
                ax3.legend(loc='upper right')
                
                if xlim is not None:
                    ax3.set_xlim(xlim)

                    
        fig.suptitle('Global Photometry Comparison with Star Labels', fontsize=18, y=0.98)
        
        plt.tight_layout(rect=[0, 0, 0.9, 1]) 
        if save_folder:
            save_name = os.path.join(save_folder, 'all_stars_comparison.png')
            plt.savefig(save_name, dpi=300, bbox_inches='tight')
            print(f" Saved comparison plot to: {save_name}")
     
        plt.show()
        
        
    # def run(self, logfile, target_rms=0.02, numstars=15, target_id, ref_star_id):
    def run(self, logfile, target_rms=0.02, numstars=12, ignor_stars = None):
        self.data = pd.DataFrame()        
        self.read_log_file(logfile=logfile, instrument="hcam")
        self.data = self.get_instrumental_mags(data=self.open_df)
        
        #========================================================================
        #remove outliner BG
        bad_sky_stars = self.filter_by_sky(self.data, tolerance=0.03)
        initial_kick = set(bad_sky_stars)
        for s_num in initial_kick:
            m_col, e_col = f'instrumag_{s_num}', f'einstrumag_{s_num}'
            self.data = self.data.drop(columns=[m_col, e_col], errors='ignore')
            print(f"Initial Filter: Drop Star {s_num} (Sky issues)")
        
        #========================================================================   
        # remove low S/N
        low_contrast_stars = self.filter_by_contrast(self.data, min_contrast=5.0)
        initial_kick = set(low_contrast_stars)
        for s_num in initial_kick:
            m_col, e_col = f'instrumag_{s_num}', f'einstrumag_{s_num}'
            self.data = self.data.drop(columns=[m_col, e_col], errors='ignore')
            print(f"Initial Filter: Drop Star {s_num} (Contrast issues)")
            
        #========================================================================
        self.df_keep = self.data.copy()
        #Drop variable stars by default
        if ignor_stars is not None:
            for t_id in ignor_stars: 
                t_id = str(t_id)
                m_col = f'instrumag_{t_id}'
                e_col = f'einstrumag_{t_id}'
                self.data = self.data.drop(columns=[m_col, e_col], errors='ignore')
        
        #========================================================================
        cleaning = True
        iteration = 1
           # --- track RMS history for plot ---
        self.rms_history = []

        while cleaning:
            mag_cols = [c for c in self.data.columns if c.startswith('instrumag_')]
            err_cols = [c for c in self.data.columns if c.startswith('einstrumag_')]
            if len(mag_cols) < 2:
                print("Less than 2 stars remaining — stopping.")
                break
                    # --- record this iteration ---
            
            
            r, self.data= self.solve_ensemble(self.data)

            star_sd = np.nanstd(r, axis=0)
            current_rms = np.sqrt(np.nanmean(r**2))
            
            self.rms_history.append(current_rms)
            
            # #rejected point by point section
            # bad_points_mask = np.abs(r) > (self.sigma_clip * star_sd[np.newaxis, :])
            # num_rejected = np.sum(bad_points_mask)
            # print(f"       =====> Iteration {iteration}: Rejected {num_rejected} points,")
            num_rejected = 0
            if num_rejected == 0:
                if current_rms <= target_rms:
                    print(f"#### RMS ({current_rms:.4f}) < {target_rms} ####")
                    cleaning = False
                    break
                    
                elif len(mag_cols) <= numstars:
                    print(f"number star {len(mag_cols)} < {numstars} ")
                    print(f"Final RMS: {current_rms:.4f}")
                    cleaning = False
                    break
                        
                else:
                    v_results = self.find_most_variable_star(self.data)
                    if v_results:
                        worst_star_col = list(v_results.keys())[0]
                        star_id = worst_star_col.split('_')[-1]

                        self.data = self.data.drop(columns=[worst_star_col, worst_star_col.replace('instrumag_', 'einstrumag_')])
                        print(f"Variable Star Rejection: Drop Star {star_id} : RMS {current_rms:.4f}")
                        
                        iteration += 1
                        continue 
            # #rejected point by point section
            # for i, (m_col, e_col) in enumerate(zip(mag_cols, err_cols)):
            #     self.data.loc[bad_points_mask[:, i], m_col] = np.nan
            #     self.data.loc[bad_points_mask[:, i], e_col] = np.nan
            
            iteration += 1
              
            if iteration > 300: # Avoid infinite loop
                print("#### Reach max iterations. ####")
                break
        self.surviving_cols = [c for c in self.data.columns if c.startswith('instrumag_')]
        self.surviving_stars = [c.split('_')[-1] for c in self.surviving_cols] #
        self.save_results()
        return 
        
    def save_results(self):

        # 1. Check data readiness
        if 'exposure_corr' not in self.data.columns:
            print("Warning: 'exposure_corr' not found. Please run solve_ensemble() first.")
            return

        # 2. Copy time and environment columns
        time_cols = [c for c in ['MJD', 'BJD', 'Exptim', 'secz'] if c in self.df_keep.columns]
        df_out = self.df_keep[time_cols].copy()
        
        # Store atmospheric correction
        df_out['exposure_corr'] = self.data['exposure_corr']

        # 3. Loop to calculate Calibrated Magnitude
        mag_cols = [c for c in self.df_keep.columns if c.startswith('instrumag_')]

        for col in mag_cols:
            star_id = col.split('_')[-1]
            err_col = f'einstrumag_{star_id}'

            # Calibrated Mag = Raw Mag - Atmosphere 
            df_out[f'mag_{star_id}'] = self.df_keep[col] - self.data['exposure_corr']

            # Copy corresponding error values
            if err_col in self.df_keep.columns:
                df_out[f'emag_{star_id}'] = self.df_keep[err_col]

        # 4. Save everything to a single file
        csv_path = os.path.join(self.save_path , 'calibrated_lightcurves.txt')
        
        with open(csv_path, 'w') as f:
            # Header: Write reference stars as a comment
            if hasattr(self, 'surviving_stars') and self.surviving_stars:
                f.write("# The following stars survived all filtering and were used as the ensemble reference:\n")
                f.write("# " + ", ".join(self.surviving_stars) + "\n")
            else:
                f.write("# Warning: No reference stars found.\n")
                
            # Table: Append pandas dataframe to the open file
            df_out.to_csv(f, index=False)
            
        print(f"Success: Saved calibrated data and reference stars to a single file: {csv_path}")

    def save_calibrated_data(self, folder_path):
        import os
        if not hasattr(self, 'real_mag_data'):
            return

        filename = "calibrated_lightcurves.csv"
        full_path = os.path.join(folder_path, filename)

        try:

            self.real_mag_data.to_csv(full_path, index=False)
            print(f"save to {full_path}")
        except Exception as e:
            print(f"error: {e}")
    
    def plot_rms_history(self, target_rms=None, save_folder=None):
        import matplotlib.pyplot as plt
        import os

        # Check if the history exists
        if not hasattr(self, 'rms_history') or not self.rms_history:
            print(" No RMS history found. Did you run the pipeline yet?")
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        
        # X-axis will just be the iteration numbers (1, 2, 3...)
        iterations = range(1, len(self.rms_history) + 1)
        
        # Plot the descending RMS curve
        ax.plot(iterations, self.rms_history, marker='o', linestyle='-', 
                color='indigo', linewidth=2, markersize=8, alpha=0.8,
                label='Ensemble RMS')
        
        # Overlay the target threshold if provided
        if target_rms is not None:
            ax.axhline(target_rms, color='crimson', linestyle='--', linewidth=2, 
                       alpha=0.8, label=f'Target RMS ({target_rms})')
            ax.legend(fontsize=12)

        # Formatting
        ax.set_title('Pipeline Convergence: Ensemble RMS per Iteration', fontsize=16, fontweight='bold')
        ax.set_xlabel('Cleaning Iteration', fontsize=12)
        ax.set_ylabel('Global RMS (mag)', fontsize=12)
        ax.grid(True, linestyle=':', alpha=0.7)
        
        # Force the X-axis to only show whole numbers (no 1.5 iterations)
        ax.set_xticks(iterations)

        # Save logic
        if save_folder:
            save_name = os.path.join(save_folder, 'rms_convergence_history.png')
            plt.savefig(save_name, dpi=300, bbox_inches='tight')
            print(f" Saved RMS history plot to: {save_name}")

        plt.tight_layout()
        plt.show()