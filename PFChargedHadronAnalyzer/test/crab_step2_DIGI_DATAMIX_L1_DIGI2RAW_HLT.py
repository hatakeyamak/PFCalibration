from WMCore.Configuration import Configuration
config = Configuration()

config.section_("General")
config.General.requestName = 'PGun_step2_DIGI_10_2_11_E0_500_PU_v1'
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
config.JobType.psetName = 'step2_DIGI_DATAMIX_L1_DIGI2RAW_HLT.py'
config.JobType.outputFiles = ['step2.root']
#config.JobType.eventsPerLumi = 2000

config.JobType.maxMemoryMB = 15000
config.JobType.numCores = 8

config.section_("Data")
config.Data.inputDBS = 'https://cmsweb.cern.ch/dbs/prod/phys03/DBSReader/'
config.Data.inputDataset = '/SinglePi/hatake-PGun_step1_GEN_SIM_10_2_11_E0_500_v1-f188f706f21b5c54b4e45dfe33c92dea/USER'
#config.Data.primaryDataset = ''
#config.Data.splitting = 'EventBased'
#config.Data.userInputFiles = open('/uscms_data/d2/hatake/PF/CMSSW_10_4_0/src/PFCalibration/PFChargedHadronAnalyzer/test/step1_file_list.txt').readlines()
config.Data.ignoreLocality = True
config.Data.splitting = 'FileBased'
config.Data.unitsPerJob = 1
NJOBS = 5000
config.Data.totalUnits = config.Data.unitsPerJob * NJOBS
config.Data.publication = False
#config.Data.publishDBS = '' default for the moment
#config.Data.outLFN = '/home/spandey/t3store/PF_PGun'
config.Data.outLFNDirBase = '/store/user/hatake/step2/PGun_step2_GEN_SIM_10_2_11_E0_500_PU_v1/'

#config.Data.outputPrimaryDataset = 'SinglePi'
config.Data.publication = True
config.Data.publishDBS = 'https://cmsweb.cern.ch/dbs/prod/phys03/DBSWriter/' # Parameter Data.publishDbsUrl has been renamed to Data.publishDBS
config.Data.outputDatasetTag = 'CMSSW_10_2_11_Step2_PU_v1' # <== Check!!!

config.section_("Site")
config.Site.storageSite = 'T3_US_Baylor'
#config.Site.blacklist = ['T3_US_UCR', 'T3_US_UMiss']
config.Site.whitelist = ['T2_*','T3_US_FNALLPC']

#config.section_("User")
#config.section_("Debug")
