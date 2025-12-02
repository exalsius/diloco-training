# Changelog

## [0.2.1](https://github.com/exalsius/diloco-training/compare/v0.2.0...v0.2.1) (2025-12-01)


### Bug Fixes

* add incremental log updates about parameter synchronization ([#63](https://github.com/exalsius/diloco-training/issues/63)) ([8b04ecd](https://github.com/exalsius/diloco-training/commit/8b04ecdd5f5205f48ef80c8e0239a10f5160c223))
* introduce detailed logging for diloco outer optimizer synchronization ([#60](https://github.com/exalsius/diloco-training/issues/60)) ([1b920b0](https://github.com/exalsius/diloco-training/commit/1b920b0fd9442b916ef839cb2ba89f42267993fa))

## [0.2.0](https://github.com/exalsius/diloco-training/compare/v0.1.0...v0.2.0) (2025-11-30)


### Features

* adjust dockerfiles to support heterogeneous amd/nvidia training ([#58](https://github.com/exalsius/diloco-training/issues/58)) ([3cd838b](https://github.com/exalsius/diloco-training/commit/3cd838b44b755173625f863aab758f9f7736cd43))

## 0.1.0 (2025-11-12)


### Features

* add a TrainingConfig model and new entrypoint script ([6bb6171](https://github.com/exalsius/diloco-training/commit/6bb61712af13514ecabe33071dc5de6c4d9fcb0d))
* add TrainingConfig model and start_training script ([3fe2d37](https://github.com/exalsius/diloco-training/commit/3fe2d37c953dd86a3d1d7592c6389508cd27f59d))
* add TrainingConfig model and start_training script ([c851033](https://github.com/exalsius/diloco-training/commit/c851033c2c01e9d6b71d471d2d69513a962005be))
* introduce model_cache_dir and dataset_cache_dir configuration ([#56](https://github.com/exalsius/diloco-training/issues/56)) ([4fbf953](https://github.com/exalsius/diloco-training/commit/4fbf953bc1de9f633dbeb59dab81ac4bd771e1f0))


### Bug Fixes

* add deleted uv.lock and install etcd ([3e94611](https://github.com/exalsius/diloco-training/commit/3e94611451f1d4744c48914c1aa17ffde35f4266))
* add latest changes from inside the pod ([948a21c](https://github.com/exalsius/diloco-training/commit/948a21c44b5d5745443b923f875a6f4b513a896f))
* add pre-commit as a dev dependency ([dd817b6](https://github.com/exalsius/diloco-training/commit/dd817b6051b4bfb8eb2114846249743fdf1933bf))
* adjust heterogenous profiling implementation with new training interface ([decd023](https://github.com/exalsius/diloco-training/commit/decd023a800d397fbac11db9121c38a64e47477f))
* adjust heterogenous training profiling with new DistributedTraining interface ([de75de4](https://github.com/exalsius/diloco-training/commit/de75de4db68480c7e2a759b00fe0f1b9a92b0a46))
* adjust main.py script with TrainingConfig ([d86aa48](https://github.com/exalsius/diloco-training/commit/d86aa48ff9a4a7cb6b6a8b3c512e7306ea3e1cb6))
* adjust main.py script with TrainingConfig ([54d9e17](https://github.com/exalsius/diloco-training/commit/54d9e170d83774c3da60981003cc016b80ec312f))
* ensure that PosixPath is serializable ([a7fedab](https://github.com/exalsius/diloco-training/commit/a7fedab4c0eec40beba7b37eec7cba591a1e9c26))
* fix logging ([31228ab](https://github.com/exalsius/diloco-training/commit/31228ab2d799453e8b065d50ca8213cdabc2ccc0))
* handle comma-separated values for experiment tags ([82d70cb](https://github.com/exalsius/diloco-training/commit/82d70cb2da6e4ce3bdb8bb337a348b9eedc228b6))
* initialize sum_local_steps in DistributedTrainer ([f97e621](https://github.com/exalsius/diloco-training/commit/f97e62154b615b48d919fba05c04622c773969d2))
* integrate missing parameters into TrainingConfig ([3071e5f](https://github.com/exalsius/diloco-training/commit/3071e5f00f86a7960ee05c0b2cafcbd420dc8167))
* introduce model_name variable initializatoin in DistributedTrainer ([358b1a2](https://github.com/exalsius/diloco-training/commit/358b1a28c3a1ec4d5fdf505661d204d8caa3d395))
* **Makefile:** Correctly declare .PHONY in Makefile ([3b18fd9](https://github.com/exalsius/diloco-training/commit/3b18fd951007bec50816152a439f12894bb68a38))
* only log to wandb if it is initialized ([e3dc41d](https://github.com/exalsius/diloco-training/commit/e3dc41de0aab9a67e292301c407378fc164563f3))
* only log to wandb if it is initialized ([1226eb0](https://github.com/exalsius/diloco-training/commit/1226eb027975744170d1f67ce741811b3998816a))
* refactor DistributedTrainer to support TrainingConfig ([e66659e](https://github.com/exalsius/diloco-training/commit/e66659e47f30525a4375604c89388d4a6300ccd7))
* refactor DistributedTrainer to support TrainingConfig ([70a1d6a](https://github.com/exalsius/diloco-training/commit/70a1d6afcc04795347c3da0507aa5fab10a73c28))
* **wandb:** show real steps on x-axis ([#57](https://github.com/exalsius/diloco-training/issues/57)) ([5c8be44](https://github.com/exalsius/diloco-training/commit/5c8be444637696535aad838f90b92336fc91ac31))


### Documentation

* add Apache 2.0 license ([2c879f1](https://github.com/exalsius/diloco-training/commit/2c879f1f843934cacb0e8a16dc2bed55274130f0))
* add example config and torchrun script ([fecc8cc](https://github.com/exalsius/diloco-training/commit/fecc8cc7fc184e175c0afac616b03bb3af3b87d9))
* add heterogenous gpu support highlight ([c00cd37](https://github.com/exalsius/diloco-training/commit/c00cd379c5c5029277c4943daf551b152cb2316c))
* add hf model upload params to example config ([303222c](https://github.com/exalsius/diloco-training/commit/303222c01138241967600854082ff35656a7c1d0))
* fix whitespaces in license ([bebd3b1](https://github.com/exalsius/diloco-training/commit/bebd3b1bbf28541ec064f004a95353d5fc9d61db))
* remove unnessary prerequiste section ([c9753ca](https://github.com/exalsius/diloco-training/commit/c9753cae4592dc0dab14592bf0562bf63a14a773))
* rewrite readme ([448c853](https://github.com/exalsius/diloco-training/commit/448c8534ea6169e34de4efc8a7583b663025d4d6))
