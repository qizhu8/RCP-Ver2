# python3 test_caseStudy.py # replaced by caseStudy_w_TCP
/Library/Frameworks/Python.framework/Versions/3.8/bin/python3 plot_testResults.py --resultFolder Results/case_study_w_TCP_TimeDiscount_alpha_2_0 --subFolderPrefix TimeDiscount --configAttributeName beta
cp Results/case_study_w_TCP_TimeDiscount_alpha_2_0/summary/system\ utility_beta.pdf Results/system\ utility-case\ study.pdf
cp Results/case_study_w_TCP_TimeDiscount_alpha_2_0/summary/system\ utility_beta.png Results/system\ utility-case\ study.png
# retransmission probability with \eta
cp Results/case_study_w_TCP_TimeDiscount_alpha_2_0/summary/retransProb_beta.pdf Results/retransProb_beta-case\ study.pdf
cp Results/case_study_w_TCP_TimeDiscount_alpha_2_0/summary/retransProb_beta.png Results/retransProb_beta-case\ study.png
# retransmission probability with time of CERCP and QRCP
cp Results/case_study_w_TCP_TimeDiscount_alpha_2_0/TimeDiscount_0_6_0/retransProb_overtime.pdf Results/retransProb_overtime.pdf
cp Results/case_study_w_TCP_TimeDiscount_alpha_2_0/TimeDiscount_0_6_0/retransProb_overtime.png Results/retransProb_overtime.png

# competing network
/Library/Frameworks/Python.framework/Versions/3.8/bin/python3 plot_testResults.py --resultFolder Results/case_study_competition_TimeDiscount_alpha_2_0 --subFolderPrefix TimeDiscount --configAttributeName beta
cp Results/case_study_competition_TimeDiscount_alpha_2_0/summary/system\ utility_beta.pdf Results/system\ utility-competition.pdf
cp Results/case_study_competition_TimeDiscount_alpha_2_0/summary/system\ utility_beta.png Results/system\ utility-competition.png


# new utility
/Library/Frameworks/Python.framework/Versions/3.8/bin/python3 plot_testResults.py --resultFolder Results/case_newutility_SumPower_alpha_2_0 --subFolderPrefix SumPower --configAttributeName beta
# utility curve
cp Results/case_newutility_SumPower_alpha_2_0/summary/system\ utility_beta.pdf Results/system\ utility-newutility.pdf
cp Results/case_newutility_SumPower_alpha_2_0/summary/system\ utility_beta.png Results/system\ utility-newutility.png

# dynamic channel
/Library/Frameworks/Python.framework/Versions/3.8/bin/python3 plot_testResults.py --resultFolder Results/dynamic_channel_TimeDiscount_alpha_2_0 --subFolderPrefix TimeDiscount --configAttributeName beta

# retransmission probability of RCP over time
cp Results/case_study_w_TCP_TimeDiscount_alpha_2_0/TimeDiscount_0_3_0/retransProb_overtime.pdf Results/change_of_retransProb_time_QRCP-dynamic\ channel.pdf
cp Results/case_study_w_TCP_TimeDiscount_alpha_2_0/TimeDiscount_0_3_0/retransProb_overtime.png Results/change_of_retransProb_time_QRCP-dynamic\ channel.png

# retransmission probability of QRCP over time for
cp Results/dynamic_channel_TimeDiscount_alpha_2_0/TimeDiscount_0_9_0/retransProb_QRCP_overtime.pdf Results/change_of_retransProb_time_QRCP-dynamic\ channel.pdf
cp Results/dynamic_channel_TimeDiscount_alpha_2_0/TimeDiscount_0_9_0/retransProb_QRCP_overtime.png Results/change_of_retransProb_time_QRCP-dynamic\ channel.png