from WMCore.Configuration import Configuration
config = Configuration()

config.section_("General")
config.General.requestName = 'PGun_step2_DIGI_10_4_0_E2_500_v1'
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
config.JobType.psetName = 'step2_DIGI_L1_DIGI2RAW_HLT.py'
config.JobType.outputFiles = ['step2.root']
#config.JobType.eventsPerLumi = 2000

config.section_("Data")
#config.Data.inputDataset = '/Single_Pion_gun_13TeV_pythia8/Fall14DR73-NoPU_MCRUN2_73_V9-v1/GEN-SIM-RAW-RECO'
#config.Data.primaryDataset = ''
#config.Data.splitting = 'EventBased'
config.Data.userInputFiles = open('/uscms_data/d2/hatake/PF/CMSSW_10_4_0/src/PFCalibration/PFChargedHadronAnalyzer/test/step1_file_list.txt').readlines()
config.Data.ignoreLocality = True
config.Data.splitting = 'FileBased'
config.Data.unitsPerJob = 1
NJOBS = 5000
config.Data.totalUnits = config.Data.unitsPerJob * NJOBS
config.Data.publication = False
#config.Data.publishDBS = '' default for the moment
#config.Data.outLFN = '/home/spandey/t3store/PF_PGun'
config.Data.outLFNDirBase = '/store/group/hcal_upgrade/hatake/step2/PGun_step1_GEN_SIM_10_4_0_E2_500_v1/'

config.section_("Site")
config.Site.storageSite = 'T3_US_FNALLPC'
#config.Site.blacklist = ['T3_US_UCR', 'T3_US_UMiss']
config.Site.whitelist = ['T2_US_*','T3_US_FNALLPC']

#config.section_("User")
#config.section_("Debug")
