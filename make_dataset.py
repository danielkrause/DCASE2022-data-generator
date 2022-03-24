import numpy as np
from db_config import DBConfig
from metadata_synthesizer import MetadataSynthesizer
from audio_synthesizer import AudioSynthesizer
from audio_mixer import AudioMixer
import pickle
from generation_parameters import get_params


params = get_params('2')

# db_config = DBConfig(params)

#LOAD DB-config which is already done
db_handler = open('db_config_fsd.obj','rb')
db_config = pickle.load(db_handler)
db_handler.close()

# noiselessSynth = MetadataSynthesizer(db_config, params, 'target_noiseless')

# mixtures_target, mixture_setup_target, foldlist_target = noiselessSynth.create_mixtures()
# # mixtures_target = np.load('actual_mixture.npy', allow_pickle=True)
# # mixtures_target = [mixtures_target, mixtures_target]
# # noiselessSynth._mixtures = mixtures_target
# metadata, stats = noiselessSynth.prepare_metadata_and_stats()
# noiselessSynth.write_metadata()
mix_target = open('mix_target_fsd.obj','rb')
mixtures_target = pickle.load(mix_target)
mix_target.close()

mix_target_set = open('mix_target_setup_fsd.obj','rb')
mixture_setup_target = pickle.load(mix_target_set)
mix_target_set.close()

# noiselessAudioSynth = AudioSynthesizer(params, mixtures_target, mixture_setup_target, db_config, 'foa')
# noiselessAudioSynth.synthesize_mixtures()
# noiselessAudioSynth = AudioSynthesizer(params, mixtures_target, mixture_setup_target, db_config, 'mic')
# noiselessAudioSynth.synthesize_mixtures()


# audioMixer = AudioMixer(params, db_config, mixtures_target, mixture_setup_target, 'foa', 'target_noisy')
# audioMixer.mix_audio()
audioMixer = AudioMixer(params, db_config, mixtures_target, mixture_setup_target, 'mic', 'target_noisy')
audioMixer.mix_audio()

params = get_params('1')

# db_config = DBConfig(params)

#LOAD DB-config which is already done
db_handler = open('db_config_nigens.obj','rb')
db_config = pickle.load(db_handler)
db_handler.close()

# noiselessSynth = MetadataSynthesizer(db_config, params, 'target_noiseless')

# mixtures_target, mixture_setup_target, foldlist_target = noiselessSynth.create_mixtures()
# # mixtures_target = np.load('actual_mixture.npy', allow_pickle=True)
# # mixtures_target = [mixtures_target, mixtures_target]
# # noiselessSynth._mixtures = mixtures_target
# metadata, stats = noiselessSynth.prepare_metadata_and_stats()
# noiselessSynth.write_metadata()
mix_target = open('mix_target_nigens.obj','rb')
mixtures_target = pickle.load(mix_target)
mix_target.close()

mix_target_set = open('mix_target_setup_nigens.obj','rb')
mixture_setup_target = pickle.load(mix_target_set)
mix_target_set.close()

noiselessAudioSynth = AudioSynthesizer(params, mixtures_target, mixture_setup_target, db_config, 'foa')
noiselessAudioSynth.synthesize_mixtures()
noiselessAudioSynth = AudioSynthesizer(mixtures_target, mixture_setup_target, db_config, 'mic')
noiselessAudioSynth.synthesize_mixtures()


audioMixer = AudioMixer(params, db_config, mixtures_target, mixture_setup_target, 'foa', 'target_noisy')
audioMixer.mix_audio()
audioMixer = AudioMixer(params, db_config, mixtures_target, mixture_setup_target, 'mic', 'target_noisy')
audioMixer.mix_audio()