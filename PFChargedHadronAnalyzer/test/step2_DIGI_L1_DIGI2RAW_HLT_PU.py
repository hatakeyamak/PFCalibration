# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: step2 --conditions auto:phase1_2018_realistic --pileup_input das:/RelValMinBias_13/CMSSW_10_3_0_pre5-103X_upgrade2018_realistic_v7-v1/GEN-SIM -n 10 --era Run2_2018 --eventcontent FEVTDEBUGHLT -s DIGI:pdigi_valid,L1,DIGI2RAW,HLT:@relval2018 --datatier GEN-SIM-DIGI-RAW --pileup AVE_50_BX_25ns --geometry DB:Extended --filein file:step1.root --fileout file:step2.root
import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras

process = cms.Process('HLT',eras.Run2_2018)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mix_POISSON_average_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Digi_cff')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.DigiToRaw_cff')
process.load('HLTrigger.Configuration.HLT_GRun_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# Input source
process.source = cms.Source("PoolSource",
    dropDescendantsOfDroppedBranches = cms.untracked.bool(False),
    fileNames = cms.untracked.vstring('root://cmseos.fnal.gov//store/user/hcal_upgrade/hatake/step1/PGun_step1_GEN_SIM_10_4_0_E2_500_v2/CRAB_PrivateMC/crab_PGun_step1_GEN_SIM_10_4_0_E2_500_v2/190110_180520/0000/step1_1.root'),
    inputCommands = cms.untracked.vstring(
        'keep *', 
        'drop *_genParticles_*_*', 
        'drop *_genParticlesForJets_*_*', 
        'drop *_kt4GenJets_*_*', 
        'drop *_kt6GenJets_*_*', 
        'drop *_iterativeCone5GenJets_*_*', 
        'drop *_ak4GenJets_*_*', 
        'drop *_ak7GenJets_*_*', 
        'drop *_ak8GenJets_*_*', 
        'drop *_ak4GenJetsNoNu_*_*', 
        'drop *_ak8GenJetsNoNu_*_*', 
        'drop *_genCandidatesForMET_*_*', 
        'drop *_genParticlesForMETAllVisible_*_*', 
        'drop *_genMetCalo_*_*', 
        'drop *_genMetCaloAndNonPrompt_*_*', 
        'drop *_genMetTrue_*_*', 
        'drop *_genMetIC5GenJs_*_*'
    ),
    secondaryFileNames = cms.untracked.vstring()
)

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('step2 nevts:10'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

process.FEVTDEBUGHLToutput = cms.OutputModule("PoolOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN-SIM-DIGI-RAW'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string('file:step2.root'),
    outputCommands = process.FEVTDEBUGHLTEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

# Additional output definition

# Other statements
process.mix.input.nbPileupEvents.averageNumber = cms.double(50.000000)
process.mix.bunchspace = cms.int32(25)
process.mix.minBunch = cms.int32(-12)
process.mix.maxBunch = cms.int32(3)
process.mix.input.fileNames = cms.untracked.vstring(['/store/relval/CMSSW_10_3_0_pre5/RelValMinBias_13/GEN-SIM/103X_upgrade2018_realistic_v7-v1/10000/F9E9DD52-4C1B-834C-AC55-C0F4BEA6E7DF.root', '/store/relval/CMSSW_10_3_0_pre5/RelValMinBias_13/GEN-SIM/103X_upgrade2018_realistic_v7-v1/10000/E71FB48E-AA82-B04E-A466-62783B73379A.root', '/store/relval/CMSSW_10_3_0_pre5/RelValMinBias_13/GEN-SIM/103X_upgrade2018_realistic_v7-v1/10000/022A4F76-7CF3-3A4F-9AD3-9708662D0CC4.root', '/store/relval/CMSSW_10_3_0_pre5/RelValMinBias_13/GEN-SIM/103X_upgrade2018_realistic_v7-v1/10000/D85A338F-CCF2-7B49-ADB1-246F67EC85D4.root', '/store/relval/CMSSW_10_3_0_pre5/RelValMinBias_13/GEN-SIM/103X_upgrade2018_realistic_v7-v1/10000/AF4BA788-237D-8540-8C51-F1D98DB52767.root', '/store/relval/CMSSW_10_3_0_pre5/RelValMinBias_13/GEN-SIM/103X_upgrade2018_realistic_v7-v1/10000/D12742E3-ED09-BB44-955F-21A001BADE7F.root'])
process.mix.digitizers = cms.PSet(process.theDigitizersValid)
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2018_realistic', '')

# Path and EndPath definitions
process.digitisation_step = cms.Path(process.pdigi_valid)
process.L1simulation_step = cms.Path(process.SimL1Emulator)
process.digi2raw_step = cms.Path(process.DigiToRaw)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.FEVTDEBUGHLToutput_step = cms.EndPath(process.FEVTDEBUGHLToutput)

# Schedule definition
process.schedule = cms.Schedule(process.digitisation_step,process.L1simulation_step,process.digi2raw_step)
process.schedule.extend(process.HLTSchedule)
process.schedule.extend([process.endjob_step,process.FEVTDEBUGHLToutput_step])
from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
associatePatAlgosToolsTask(process)

# customisation of the process.

# Automatic addition of the customisation function from HLTrigger.Configuration.customizeHLTforMC
from HLTrigger.Configuration.customizeHLTforMC import customizeHLTforMC 

#call to customisation function customizeHLTforMC imported from HLTrigger.Configuration.customizeHLTforMC
process = customizeHLTforMC(process)

# End of customisation functions

# Customisation from command line

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion

#process.options.numberOfThreads = cms.untracked.uint32(2)
