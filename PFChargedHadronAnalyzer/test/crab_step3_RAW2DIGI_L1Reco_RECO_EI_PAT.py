from WMCore.Configuration import Configuration
config = Configuration()

config.section_("General")
#config.General.requestName = 'PGun_step3_RECO_10_2_11_E0_500_v1'
config.General.requestName = 'PGun_step3_RECO_10_2_11_E0_500_PU_v1'
config.General.workArea = 'crab_projects'

#optional
#config.General.transferOutputs
#config.General.transferLogs
#config.General.failureLimit = 

#Expert use
#config.General.instance
#config.General.activity

config.section_("JobType")
config.JobType.pluginName = 'Analysis'
#config.JobType.psetName = 'step3_RAW2DIGI_L1Reco_RECO_EI_PAT_VALIDATION_DQM.py'
#config.JobType.psetName = 'step3_RAW2DIGI_L1Reco_RECO_RECOSIM_EI_PAT_VALIDATION_DQM.py'
config.JobType.psetName = 'step3_RAW2DIGI_L1Reco_RECO_RECOSIM_EI_PAT.py'
config.JobType.outputFiles = ['step3.root']
#config.JobType.eventsPerLumi = 2000

config.JobType.maxMemoryMB = 3000

config.section_("Data")
#config.Data.inputDBS = 'https://cmsweb.cern.ch/dbs/prod/phys03/DBSReader/'
#config.Data.inputDataset = '/SinglePi/hatake-CMSSW_10_4_0_Step2_v3-6cb52d4242603d1fbf0661ab850234de/USER'
#config.Data.inputDataset = '/Single_Pion_gun_13TeV_pythia8/Fall14DR73-NoPU_MCRUN2_73_V9-v1/GEN-SIM-RAW-RECO'
#config.Data.primaryDataset = ''
#config.Data.splitting = 'Automatic'
config.Data.userInputFiles = open('/home/hatake/ana_cms/PF/CMSSW_10_2_11/src/PFCalibration/PFChargedHadronAnalyzer/test/step2_file_list_v2.txt').readlines()
config.Data.ignoreLocality = True
config.Data.splitting = 'FileBased'
config.Data.unitsPerJob = 1
NJOBS = 5000
#NJOBS = 20000
config.Data.totalUnits = config.Data.unitsPerJob * NJOBS
#config.Data.publication = False
#config.Data.publishDBS = '' default for the moment
#config.Data.outLFN = '/home/spandey/t3store/PF_PGun'
#config.Data.outLFNDirBase = '/store/user/spandey/step3/PGun_step3_RECO_1002_2_200_Feb_13/'
#config.Data.outLFNDirBase = '/store/user/hatake/step3/PGun_step3_RECO_10_2_11_E0_500_v1/'
config.Data.outLFNDirBase = '/store/user/hatake/step3/PGun_step3_RECO_10_2_11_E0_500_PU_v1/'

config.Data.outputPrimaryDataset = 'SinglePi'
config.Data.publication = True
config.Data.publishDBS = 'https://cmsweb.cern.ch/dbs/prod/phys03/DBSWriter/' # Parameter Data.publishDbsUrl has been renamed to Data.publishDBS
config.Data.outputDatasetTag = config.General.requestName # <== Check!!!

config.section_("Site")
config.Site.storageSite = 'T3_US_Baylor'
#config.Site.blacklist = ['T3_US_UCR', 'T3_US_UMiss']
config.Site.whitelist = ['T2_US_*','T3_US_FNALLPC']

#config.section_("User")
#config.section_("Debug")
