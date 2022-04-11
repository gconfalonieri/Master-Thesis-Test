import matplotlib.pyplot as plt

def get_labels_pie_plot(df_label):

    for i in df_label.index:
        n_total = df_label['N_COGNITIVE_EFFORT'][i] + df_label['N_NOT_COGNITIVE_EFFORT'][i]
        perc_cognitive_effort =  round(df_label['N_COGNITIVE_EFFORT'][i] / n_total, 2)
        perc_not_cognitive_effort = round(df_label['N_NOT_COGNITIVE_EFFORT'][i] / n_total, 2)
        labels = ['Cognitive Effort (' + str(perc_cognitive_effort) + ')' , 'Not Cognitive Effort (' + str(perc_not_cognitive_effort) + ')']
        data = [df_label['N_COGNITIVE_EFFORT'][i], df_label['N_NOT_COGNITIVE_EFFORT'][i]]
        plt.figure()
        plt.pie(data, labels=labels)
        plt.savefig('plots/median_labels/macro_categories/' + df_label['MACRO_CATEGORY'][i] + '.png')
        plt.close()


def get_total_labels_pie_plot(df_label):

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
    plt.savefig('plots/median_labels/macro_categories/all.png')
    plt.close()