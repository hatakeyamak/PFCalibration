from WMCore.Configuration import Configuration
config = Configuration()

config.section_("General")
config.General.requestName = 'PGun_step1_GEN_SIM_10_2_11_E0_500_v2'
config.General.workArea = 'crab_projects'

#optional
#config.General.transferOutputs
#config.General.transferLogs
#config.General.failureLimit = 

#Expert use
#config.General.instance
#config.General.activity

config.section_("JobType")
config.JobType.pluginName = 'PrivateMC'
#config.JobType.psetName = 'SinglePiE50HCAL_pythia8_cfi_GEN_SIM.py'
config.JobType.psetName = 'SinglePi0E10_pythia8_cfi_GEN_SIM.py'
config.JobType.outputFiles = ['step1.root']
#config.JobType.eventsPerLumi = 2000
config.JobType.eventsPerLumi = 500

config.section_("Data")
#config.Data.inputDataset = '/Single_Pion_gun_13TeV_pythia8/Fall14DR73-NoPU_MCRUN2_73_V9-v1/GEN-SIM-RAW-RECO'
#config.Data.primaryDataset = ''
config.Data.splitting = 'EventBased'
#config.Data.unitsPerJob = 2000
config.Data.unitsPerJob = 1000
NJOBS = 10000
config.Data.totalUnits = config.Data.unitsPerJob * NJOBS
config.Data.publication = False
#config.Data.publishDBS = '' default for the moment
#config.Data.outLFN = '/home/spandey/t3store/PF_PGun'
config.Data.outLFNDirBase = '/store/user/hatake/step1/PGun_step1_GEN_SIM_10_2_11_E0_500_v2/'

config.Data.outputPrimaryDataset = 'SinglePi'
config.Data.publication = True
config.Data.publishDBS = 'https://cmsweb.cern.ch/dbs/prod/phys03/DBSWriter/' # Parameter Data.publishDbsUrl has been renamed to Data.publishDBS
config.Data.outputDatasetTag = config.General.requestName # <== Check!!!

config.section_("Site")
config.Site.storageSite = 'T3_US_Baylor'
#config.Site.blacklist = ['T3_US_UCR', 'T3_US_UMiss']
#config.Site.whitelist = ['T2_CH_CERN','T2_KR_KNU']

#config.section_("User")
#config.section_("Debug")
