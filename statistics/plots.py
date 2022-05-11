from pprint import pprint

import matplotlib.pyplot as plt
import numpy
import numpy as np
from sklearn import preprocessing

def get_error_bar_plot_time_questions(df_questions_statistics, cunks, type):

    list_average_times = df_questions_statistics['AVERAGE_TIME']
    list_standard_deviations = df_questions_statistics['STANDARD_DEVIATION']

    sublists_average_times = numpy.split(list_average_times, cunks)
    sublists_standard_deviations = numpy.split(list_standard_deviations, cunks)

    for i in range(0, cunks):
        fig, ax = plt.subplots()
        barWidth = 0.25
        br = np.arange(len(sublists_average_times[i]))
        ax.bar(br, sublists_average_times[i], yerr=sublists_standard_deviations[i], width=barWidth, edgecolor='grey')
        ax.set_title('Answer Time for '+ type + ' - ' + str(i+1))
        ax.set_xlabel(type)
        ax.set_ylabel('Average Answer Time (s)')
        plt.savefig('plots/statistics_times_for_' + type.lower() +'_' + str(i+1) + '.png')
        plt.close()

def get_error_bar_plot_time_questions_normalized(df_questions_statistics, chunks, type):

    list_average_times = df_questions_statistics['AVERAGE_TIME']
    list_standard_deviations = df_questions_statistics['STANDARD_DEVIATION']

    normalized_average_times = []
    normalized_standard_deviations = []

    for x in list_average_times:
        x_norm = (x - min(list_average_times)) / ( max(list_average_times) - min(list_average_times) )
        normalized_average_times.append(x_norm)

    for x in list_standard_deviations:
        x_norm = (x - min(list_standard_deviations)) / ( max(list_standard_deviations) - min(list_standard_deviations) )
        normalized_standard_deviations.append(x_norm)

    sublists_average_times = numpy.split(numpy.array(normalized_average_times), chunks)
    sublists_standard_deviations = numpy.split(numpy.array(normalized_standard_deviations), chunks)

    for i in range(0, chunks):
        fig, ax = plt.subplots()
        barWidth = 0.25
        br = np.arange(len(sublists_average_times[i]))
        ax.bar(br, sublists_average_times[i], yerr=sublists_standard_deviations[i], width=barWidth, edgecolor='grey')
        ax.set_title('Normalized Average Answer Time for '+ type + ' - ' + str(i+1))
        ax.set_xlabel(type)
        ax.set_ylabel('Normalized Answer Time')
        plt.savefig('plots/statistics_times_for_'+ type.lower() +'_normalized_' + str(i+1) + '.png')
        plt.close()


def get_questions_statistics_bar_plot(df_questions_statistics):

    list_rights = []
    list_wrongs = []

    for i in df_questions_statistics.index:
        n_right = (df_questions_statistics['RIGHT'][i] / df_questions_statistics['TOTAL'][i]) * 100
        n_wrong = (df_questions_statistics['WRONG'][i] / df_questions_statistics['TOTAL'][i]) * 100
        list_rights.append(n_right)
        list_wrongs.append(n_wrong)

    fig, ax = plt.subplots()
    barWidth = 0.25
    br1 = np.arange(len(list_rights))
    br2 = [x + barWidth for x in br1]

    ax.set_title("Righ or Wrong Answers")
    ax.set_xlabel("Questions")
    ax.set_ylabel("Counters")

    ax.bar(br1, list_rights, color='g', width=barWidth, edgecolor='grey', label='RIGHT')
    ax.bar(br2, list_wrongs, color='r', width=barWidth, edgecolor='grey', label='WRONG')

    plt.legend()
    plt.savefig('plots/questions_statistics.png')
    plt.close()


def get_answers_times_statistics_bar_plot(df_questions_statistics, type):

    list_times = df_questions_statistics['AVERAGE_TIME']
    st_dev = df_questions_statistics['STANDARD_DEVIATION']

    plt.figure()
    barWidth = 0.25
    br = np.arange(len(list_times))

    fig, ax = plt.subplots()
    ax.set_title('Answer Time for ' + type + ' - Total')
    ax.set_xlabel(type)
    ax.set_ylabel('Answer Time (s)')
    plt.bar(br, list_times, yerr=st_dev, width=barWidth, edgecolor='grey')
    plt.savefig('plots/average_times_for_'+ type.lower() +'.png')
    plt.close()

def get_answers_times_statistics_bar_plot_normalized(df_questions_statistics, type):

    list_average_times = df_questions_statistics['AVERAGE_TIME']
    list_standard_deviations = df_questions_statistics['STANDARD_DEVIATION']

    normalized_average_times = []
    normalized_standard_deviations = []

    for x in list_average_times:
        x_norm = (x - min(list_average_times)) / ( max(list_average_times) - min(list_average_times) )
        normalized_average_times.append(x_norm)

    for x in list_standard_deviations:
        x_norm = (x - min(list_standard_deviations)) / ( max(list_standard_deviations) - min(list_standard_deviations) )
        normalized_standard_deviations.append(x_norm)

    plt.figure()
    barWidth = 0.25
    br = np.arange(len(list_average_times))

    fig, ax = plt.subplots()
    ax.set_title('Normalized Answer Time for ' + type + ' - Total')
    ax.set_xlabel(type)
    ax.set_ylabel('Normalized Answer Time')
    plt.bar(br, normalized_average_times, yerr=normalized_standard_deviations, width=barWidth, edgecolor='grey')
    plt.savefig('plots/normalized_average_times_for_' + type.lower() + '.png')
    plt.close()

def get_labels_pie_plot(df_label, label_column, path_init):

    for i in df_label.index:
        n_total = df_label['N_COGNITIVE_EFFORT'][i] + df_label['N_NOT_COGNITIVE_EFFORT'][i]
        perc_cognitive_effort =  round(df_label['N_COGNITIVE_EFFORT'][i] / n_total, 2)
        perc_not_cognitive_effort = round(df_label['N_NOT_COGNITIVE_EFFORT'][i] / n_total, 2)
        labels = ['Cognitive Effort (' + str(perc_cognitive_effort) + ')' , 'Not Cognitive Effort (' + str(perc_not_cognitive_effort) + ')']
        data = [df_label['N_COGNITIVE_EFFORT'][i], df_label['N_NOT_COGNITIVE_EFFORT'][i]]
        fig, ax = plt.subplots()
        ax.set_title(label_column + " - " + df_label[label_column][i])
        plt.pie(data, labels=labels)
        plt.savefig(path_init + df_label[label_column][i] + '.png')
        plt.close()


def get_total_labels_pie_plot(df_label, label_column, path_init):

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
    fig, ax = plt.subplots()
    ax.set_title(label_column + " - ALL")
    plt.pie(data, labels=labels)
    plt.savefig( path_init + 'label_statistics_all.png')
    plt.close()