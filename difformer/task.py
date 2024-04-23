import numpy as np

import torch

from fairseq import metrics
from fairseq.tasks import LegacyFairseqTask, register_task
from fairseq.tasks.translation import TranslationTask
from fairseq.tasks.translation_lev import TranslationLevenshteinTask

from .generator import DifformerGenerator


@register_task("difformer")
class DifformerTask(TranslationLevenshteinTask):
    @staticmethod
    def add_args(parser):
        TranslationLevenshteinTask.add_args(parser)

        parser.add_argument(
            "--decoding-steps",
            type=int, metavar="N", default=2000,
            help="Decoding steps"
        )
        parser.add_argument(
            "--decoding-fixed-t",
            type=float, metavar="D",
            help="Fix the t fed to the model to D. Range: [0, 1)"
        )
        parser.add_argument(
            "--decoding-early-stopping",
            type=int, metavar="N", default=0,
            help="Stop decoding N steps earlier"
        )

        parser.add_argument(
            "--decoding-noise-schedule",
            type=str, metavar="STR",
            help="The noise schedule during decoding"
        )
        parser.add_argument(
            "--decoding-rescaling-factor",
            type=float, metavar="D", default=1.0,
            help="The rescaling factor during decoding"
        )
        parser.add_argument(
            "--decoding-vp-rf",
            action="store_true",
            help="Use the variance-preserving rescaling factor during decoding"
        )

        parser.add_argument(
            "--clamping",
            action="store_true",
            help="Use clamping trick during decoding"
        )

        parser.add_argument(
            "--length-beam-size",
            type=int, metavar="N", default=1,
            help="Decode with N different lengths candidates"
        )
        parser.add_argument(
            "--noise-beam-size",
            type=int, metavar="N", default=1,
            help="For each length candidate, generate N samples"
        )
        
        parser.add_argument(
            "--ppl-mbr",
            action="store_true",
            help="Apply minimum bayes risk (MBR) of PPL to pick the best candidate sample"
        )
        parser.add_argument(
            "--bleu-mbr",
            action="store_true",
            help="Apply minimum bayes risk (MBR) of BLEU to pick the best candidate sample"
        )

        parser.add_argument(
            "--retain-z-0-hat",
            action="store_true",
            help="When retain history, $z_0$ is retained rather than $z_t$"
        )

    @staticmethod
    def base_args(args):
        args.decoding_steps = getattr(args, "decoding_steps", 2000)
        args.decoding_fixed_t = getattr(args, "decoding_fixed_t", None)
        args.early_stopping = getattr(args, "decoding_early_stopping", 0)

        args.decoding_noise_schedule = getattr(args, "decoding_noise_schedule", None)
        args.decoding_rescaling_factor = getattr(args, "decoding_rescaling_factor", 1.0)
        args.decoding_vp_rf = getattr(args, "decoding_vp_rf", 1.0)

        args.clamping = getattr(args, "clamping", False)

        args.length_beam_size = getattr(args, "length_beam_size", 1)
        args.noise_beam_size = getattr(args, "noise_beam_size", 1)
        args.beam = args.length_beam_size * args.noise_beam_size

        args.ppl_mbr = getattr(args, "ppl_mbr", False)
        args.bleu_mbr = getattr(args, "bleu_mbr", False)

        args.retain_iter_history = getattr(args, "retain_iter_history", False)
        args.retain_z_0_hat = getattr(args, "retain_z_0_hat", False)

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)
        self.base_args(args)

        assert (
            args.decoding_fixed_t is None
            or 0 <= args.decoding_fixed_t < 1
        ), (
            "The fixed t must be in [0, 1)"
        )

        assert not (args.ppl_mbr and args.bleu_mbr), (
            "You can not apply both MBR of PPL and BLEU"
        )

        args.early_stopping = max(args.early_stopping, 0)
        self.args = args

    def build_model(self, args):
        args.decoding_steps = self.args.decoding_steps
        args.decoding_fixed_t = self.args.decoding_fixed_t

        args.decoding_noise_schedule = self.args.decoding_noise_schedule
        args.decoding_rescaling_factor = self.args.decoding_rescaling_factor
        args.decoding_vp_rf = self.args.decoding_vp_rf

        args.clamping = self.args.clamping

        return super().build_model(args)

    def build_generator(self, *_, **__):
        return DifformerGenerator(
            tgt_dict=self.target_dictionary,
            steps=self.args.decoding_steps,
            early_stopping=self.args.early_stopping,
            length_beam_size=self.args.length_beam_size,
            noise_beam_size=self.args.noise_beam_size,
            ppl_mbr=self.args.ppl_mbr,
            bleu_mbr=self.args.bleu_mbr,
            retain_history=self.args.retain_iter_history,
            retain_z_0_hat=self.args.retain_z_0_hat,
        )

    def valid_step(self, sample, model, criterion):
        sample["prev_target"] = sample["target"]

        # enable evaluate bleu during training
        return TranslationTask.valid_step(self, sample, model, criterion)
    
    # override this method in order to fix bugs
    def reduce_metrics(self, logging_outputs, criterion):
        LegacyFairseqTask.reduce_metrics(self, logging_outputs, criterion)
        if self.args.eval_bleu:

            def sum_logs(key):
                result = sum(log.get(key, 0) for log in logging_outputs)
                # fix multi-gpu bugs
                return result.cpu() if torch.is_tensor(result) else result

            counts, totals = [], []
            for i in range(4):
                counts.append(sum_logs("_bleu_counts_" + str(i)))
                totals.append(sum_logs("_bleu_totals_" + str(i)))

            if max(totals) > 0:
                # log counts as numpy arrays -- log_scalar will sum them correctly
                metrics.log_scalar("_bleu_counts", np.array(counts))
                metrics.log_scalar("_bleu_totals", np.array(totals))
                metrics.log_scalar("_bleu_sys_len", sum_logs("_bleu_sys_len"))
                metrics.log_scalar("_bleu_ref_len", sum_logs("_bleu_ref_len"))

                def compute_bleu(meters):
                    import inspect
                    import sacrebleu

                    fn_sig = inspect.getfullargspec(sacrebleu.compute_bleu)[0]
                    if "smooth_method" in fn_sig:
                        smooth = {"smooth_method": "exp"}
                    else:
                        smooth = {"smooth": "exp"}
                    bleu = sacrebleu.compute_bleu(
                        correct=meters["_bleu_counts"].sum,
                        total=meters["_bleu_totals"].sum,
                        sys_len=meters["_bleu_sys_len"].sum,
                        ref_len=meters["_bleu_ref_len"].sum,
                        **smooth
                    )
                    return round(bleu.score, 2)

                metrics.log_derived("bleu", compute_bleu)

            # avoid KeyError when output empty sentences
            else:
                metrics.log_derived("bleu", lambda _: 0.0)
