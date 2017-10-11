# get data from the spatialite database
import sqlite3
#for plotting
import seaborn as sns
import matplotlib.pyplot as plt
#for data preprocessing
import pandas as pd
from GaussianMixtrueModel import GaussianMixture

db_name = "../Goali/OSM_database/data/USA_osm.db"

conn = sqlite3.connect(db_name)
conn.enable_load_extension(True)
conn.load_extension("mod_spatialite")

query = """
        SELECT ac.HOUR, ac.MONTH, ac.COUNTY, ac.DAY_WEEK, ac.ROAD_FNC, ac.ROUTE
        FROM Fatal_Motor_Vehicle_Accidents as ac
        WHERE ac.HOUR<99
        """
accident_data = pd.read_sql(query, conn)
accident_data.head(5)

# fit accident time distribution data
data = accident_data.HOUR

# Find best Mixture Gaussian model
n_iterations = 20
n_random_restarts = 500
best_mix = None
best_loglike = float('-inf')
print('Computing best model with random restarts...\n')
for _ in range(n_random_restarts):
    mix = GaussianMixture(data)
    for _ in range(n_iterations):
        try:
            mix.iterate(verbose=True)
            if mix.loglike > best_loglike:
                best_loglike = mix.loglike
                best_mix = mix
        except (ZeroDivisionError, ValueError, RuntimeWarning): # Catch division errors from bad starts, and just throw them out...
            pass

print('\n\nDone. 🙂')
print(best_mix)

# plot data and fixed GMM model
sns.distplot(data, bins=50, kde=False, norm_hist=True)
g_both = [best_mix.pdf(e) for e in x]
plt.plot(x, g_both, label='gaussian mixture');
plt.legend();