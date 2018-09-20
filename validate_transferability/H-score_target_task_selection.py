"""
This file is for showing the relationship between transferability and transfer performance under the scenario of target task selection.
"""
import numpy as np
import matplotlib.pyplot as plt

score_p = np.load('../outputs/score_p.npy')
trn_loss_p = np.load('../outputs/trn_loss_p.npy')
trn_accs_p = np.load('../outputs/trn_accs_p_.npy')
tst_accs_p = np.load('../outputs/tst_accs_p.npy')

score_a = np.load('../outputs/score_a.npy')
trn_loss_a = np.load('../outputs/trn_loss_a.npy')
trn_accs_a = np.load('../outputs/trn_accs_a.npy')
tst_accs_a = np.load('../outputs/tst_accs_a.npy')

indices = score_a>=score_p
score_p = score_p[indices]
score_a = score_a[indices]
trn_loss_p = trn_loss_p[indices]
tst_accs_p = tst_accs_p[indices]
trn_accs_p = trn_accs_p[indices]

score_p_r = np.divide(score_p, score_a)

indices = np.argsort(score_p)
plt.figure()
plt.plot(score_p[indices], tst_accs_p[indices])
plt.xlabel('transferability')
plt.ylabel('tst acc')
plt.title('tst acc vs unnormalized H-score')
plt.show()

plt.figure()
plt.plot(score_p[indices], trn_loss_p[indices])
plt.xlabel('transferability')
plt.ylabel('log-loss')
plt.title('log-loss vs unnormalized H-score')
plt.show()

indices = np.argsort(score_p_r)
plt.figure()
plt.plot(score_p_r[indices], trn_accs_p[indices], 'b-', marker='o', label = 'training accuracy')
plt.plot(score_p_r[indices], tst_accs_p[indices], 'r-', marker='o', label='testing accuracy')
plt.xlabel('transferability')
plt.ylabel('accuracy')
plt.legend(loc='lower right')
plt.show()

plt.figure()
plt.plot(score_p_r[indices], trn_loss_p[indices], 'b-', marker='o')
plt.xlabel('transferability')
plt.ylabel('log-loss')
plt.show()

plt.rcParams.update({'font.size': 12})
fig, ax1 = plt.subplots()
ax1.plot(score_p_r[indices], trn_accs_p[indices], 'r-', marker='o', label = 'training accuracy')
ax1.plot(score_p_r[indices], tst_accs_p[indices], 'g-', marker='o', label='testing accuracy')
ax1.set_xlabel('transferability')
# Make the y-axis label, ticks and tick labels match the line color.
ax1.set_ylabel('accuracy')
ax1.tick_params('y')

ax2 = ax1.twinx()
ax2.plot(score_p_r[indices], trn_loss_p[indices], 'b-', marker='o', label='log-loss')
ax2.set_ylabel('log-loss')
ax2.tick_params('y')

fig.tight_layout()
fig.legend(bbox_to_anchor=(0.15, 0.9), loc=2, borderaxespad=0.)
plt.show()
