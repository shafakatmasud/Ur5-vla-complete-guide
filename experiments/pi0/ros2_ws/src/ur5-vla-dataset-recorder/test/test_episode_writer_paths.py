import numpy as np
import pytest
import yaml
from ur5_vla_dataset_recorder.episode_writer import EpisodeWriter


@pytest.mark.parametrize('subdirectory',['staging/episodes','staging/experiment_2_sort_fruits/episodes'])
def test_successful_frame_transaction_uses_selected_episode_directory(tmp_path,subdirectory):
    writer=EpisodeWriter(tmp_path,90,4,episode_subdirectory=subdirectory)
    temp=writer.start('episode_test',{'task':'Sort the fruits'})
    assert temp.parent==tmp_path/subdirectory
    writer.enqueue({'frame_index':0},np.zeros((8,8,3),dtype=np.uint8),np.zeros((8,8,3),dtype=np.uint8))
    writer.stop();final=writer.finalize('success',{})
    assert final==tmp_path/subdirectory/'episode_test'
    assert yaml.safe_load((final/'metadata.yaml').read_text())['frames']==1
    assert (final/'frames.jsonl').exists()


@pytest.mark.parametrize('subdirectory',['','.', '../outside','/tmp/outside'])
def test_episode_directory_cannot_escape_dataset_root(tmp_path,subdirectory):
    writer=EpisodeWriter(tmp_path,90,4,episode_subdirectory=subdirectory)
    with pytest.raises(ValueError):writer.start('episode_test',{})


def test_symlink_escape_is_rejected(tmp_path):
    root=tmp_path/'dataset';root.mkdir();outside=tmp_path/'outside';outside.mkdir()
    (root/'link').symlink_to(outside,target_is_directory=True)
    with pytest.raises(ValueError):EpisodeWriter(root,90,4,episode_subdirectory='link/episodes').start('episode_test',{})
