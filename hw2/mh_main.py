# after we find the optimal setting
import numpy as np
import matplotlib.pyplot as plt

home = np.load('result/sigma_0.005_t_50_home_mc.npy')
strength = np.load('result/sigma_0.005_t_50_theta.npy')

plt.hist(home, bins=60)
plt.title('Posterior histogram of home')
# plt.show()
plt.savefig('result/hist.png', format='png')
plt.close()

strength_mean = np.mean([strength[i, 1] for i in range(strength.shape[0])], axis=0)
plt.plot(strength_mean[0,:], strength_mean[1,:], 'o')
plt.title('Estimated attacking and defending strength')
plt.ylabel('defending strength')
plt.xlabel('attacking strength')
# plt.text(1, -0.6, r'$\sum_{i=0}^\infty x_i$', fontsize=20)
team_name = ['Arsenal', 'Aston Villa', 'Cardiff City', 'Chelsea', 'Crystal Palace',
             'Everton','Fulham','Hull City','Liverpool','Manchester City',
             'Manchester United','Newcastle United','Norwich City','Southampton','Stoke City',
             'Sunderland','Swansea City','Tottenham Hotspurs','West Bromwich Albion','West Ham United']
for i in range(20):
    plt.text(strength_mean[0, i], strength_mean[1, i], team_name[i], fontsize=6)
plt.savefig('result/strength.png', format='png', dpi=600)
plt.close()