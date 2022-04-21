import matplotlib.pyplot as plt
import numpy as np


def get_questiosn_statistics_bar_plot(df_questions_statistics):

    list_rights = []
    list_wrongs = []

    for i in df_questions_statistics.index:
        n_right = (df_questions_statistics['RIGHT'][i] / df_questions_statistics['TOTAL'][i]) * 100
        n_wrong = (df_questions_statistics['WRONG'][i] / df_questions_statistics['TOTAL'][i]) * 100
        list_rights.append(n_right)
        list_wrongs.append(n_wrong)

    barWidth = 0.25
    br1 = np.arange(len(list_rights))
    br2 = [x + barWidth for x in br1]

    plt.bar(br1, list_rights, color='g', width=barWidth, edgecolor='grey', label='RIGHT')
    plt.bar(br2, list_wrongs, color='r', width=barWidth, edgecolor='grey', label='WRONG')

    plt.legend()
    plt.savefig('plots/questions_statistics.png')


def get_labels_pie_plot(df_label, label_column, path_init):

    for i in df_label.index:
        n_total = df_label['N_COGNITIVE_EFFORT'][i] + df_label['N_NOT_COGNITIVE_EFFORT'][i]
        perc_cognitive_effort =  round(df_label['N_COGNITIVE_EFFORT'][i] / n_total, 2)
        perc_not_cognitive_effort = round(df_label['N_NOT_COGNITIVE_EFFORT'][i] / n_total, 2)
        labels = ['Cognitive Effort (' + str(perc_cognitive_effort) + ')' , 'Not Cognitive Effort (' + str(perc_not_cognitive_effort) + ')']
        data = [df_label['N_COGNITIVE_EFFORT'][i], df_label['N_NOT_COGNITIVE_EFFORT'][i]]
        plt.figure()
        plt.pie(data, labels=labels)
        plt.savefig(path_init + df_label[label_column][i] + '.png')
        plt.close()


def get_total_labels_pie_plot(df_label, path_init):

    cognitive_effort = 0
    not_cognitive_effort = 0

    for i in df_label.index:
        cognitive_effort += df_label['N_COGNITIVE_EFFORT'][i]
        not_cognitive_effort += df_label['N_NOT_COGNITIVE_EFFORT'][i]

    total = cognitive_effort + not_cognitive_effort

    perc_cognitive_effort =  round(cognitive_effort / total, 2)
    perc_not_cognitive_effort = round(not_cognitive_effort / total, 2)
    labels = ['Cognitive Effort (' + str(perc_cognitive_effort) + ')' , 'Not Cognitive Effort (' + str(perc_not_cognitive_effort) + ')']
    data = [cognitive_effort, not_cognitive_effort]
    plt.figure()
    plt.pie(data, labels=labels)
    plt.savefig( path_init + 'all.png')
    plt.close()