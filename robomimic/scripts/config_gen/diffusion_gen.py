from robomimic.scripts.config_gen.helper import *
import os

def make_generator_helper(args):
    algo_name_short = "diffusion_policy"

    generator = get_generator(
        algo_name="diffusion_policy",
        config_file=os.path.join(base_path, 'robomimic/exps/templates/diffusion_policy.json'),
        args=args,
        algo_name_short=algo_name_short,
        pt=True,
    )
    if args.ckpt_mode is None:
        args.ckpt_mode = "off"

    generator.add_param(
        key="train.num_data_workers",
        name="",
        group=-1,
        values=[8],
    )

    generator.add_param(
        key="train.num_epochs",
        name="",
        group=-1,
        values=[1000],
    )

    generator.add_param(
        key="train.data_format",
        name="df",
        group=1123,
        values=[args.data_format],
        value_names=[
            "data_format",
        ]
    )

    # use ddim by default
    generator.add_param(
        key="algo.ddim.enabled",
        name="ddim",
        group=1001,
        values=[
            True,
            # False,
        ],
        hidename=True,
    )
    generator.add_param(
        key="algo.ddpm.enabled",
        name="ddpm",
        group=1001,
        values=[
            False,
            # True,
        ],
        hidename=True,
    )

    if args.env == "r2d2":
        generator.add_param(
            key="train.data_format",
            name="df",
            group=1123,
            values=[args.data_format],
            value_names=[
                "data_format",
            ]
        )
        if args.data_format == 'hdf5':
            generator.add_param(
                key="train.data",
                name="ds",
                group=2,
                values=[
                    [{"path": p} for p in scan_datasets("/ssd1/datasets/robotics/r2d2/tasks/pen/hdf5", postfix="trajectory_im128.h5")],
                ],
                value_names=[
                    "pen-in-cup",
                ],
            )
            generator.add_param(
                key="train.action_keys",
                name="ac_keys",
                group=-1,
                values=[
                    [
                        "action/abs_pos",
                        "action/abs_rot_6d",
                        "action/gripper_position",
                    ],
                ],
                value_names=[
                    "abs",
                ],
                hidename=True,
            )
            generator.add_param(
                key="observation.modalities.obs.rgb",
                name="cams",
                group=130,
                values=[
                    # ["camera/image/hand_camera_left_image"],
                    # ["camera/image/hand_camera_left_image", "camera/image/hand_camera_right_image"],
                    ["camera/image/hand_camera_left_image", "camera/image/varied_camera_1_left_image", "camera/image/varied_camera_2_left_image"],
                    # [
                    #     "camera/image/hand_camera_left_image", "camera/image/hand_camera_right_image",
                    #     "camera/image/varied_camera_1_left_image", "camera/image/varied_camera_1_right_image",
                    #     "camera/image/varied_camera_2_left_image", "camera/image/varied_camera_2_right_image",
                    # ],
                ],
                value_names=[
                    # "wrist",
                    # "wrist-stereo",
                    "3cams",
                    # "3cams-stereo",
                ]
            )
            generator.add_param(
                key="observation.modalities.obs.low_dim",
                name="ldkeys",
                group=2498,
                values=[
                    ["robot_state/cartesian_position", "robot_state/gripper_position"],
                    # [
                    #     "robot_state/cartesian_position", "robot_state/gripper_position",
                    #     "camera/extrinsics/hand_camera_left", "camera/extrinsics/hand_camera_left_gripper_offset",
                    #     "camera/extrinsics/hand_camera_right", "camera/extrinsics/hand_camera_right_gripper_offset",
                    #     "camera/extrinsics/varied_camera_1_left", "camera/extrinsics/varied_camera_1_right",
                    #     "camera/extrinsics/varied_camera_2_left", "camera/extrinsics/varied_camera_2_right",
                    # ]
                ],
                value_names=[
                    "proprio",
                    # "proprio-extrinsics",
                ]
            )


        elif args.data_format == 'rlds':
            generator.add_param(
                key="train.data",
                name="ds",
                group=2,
                values=[
                    [
                        {"path": "/iris/u/jyang27/rlds_data",
                       "name": "r2_d2"}, # replace with your own path
                    ],
                ],
                value_names=[
                    "pen-in-cup"
                ],
            )
            generator.add_param(
                key="train.action_keys",
                name="ac_keys",
                group=-1,
                values=[
                    [
                        "action_dict/abs_pos",
                        "action_dict/abs_rot_6d",
                        "action_dict/gripper_position",
                    ],
                ],
                value_names=[
                    "abs",
                ],
                hidename=True,
            )
            generator.add_param(
                key="observation.modalities.obs.rgb",
                name="cams",
                group=130,
                values=[
                    # ["camera/image/hand_camera_left_image"],
                    # ["camera/image/hand_camera_left_image", "camera/image/hand_camera_right_image"],
                    ["wrist_image_left", "exterior_image_1_left", "exterior_image_2_left"],
                    # [
                    #     "camera/image/hand_camera_left_image", "camera/image/hand_camera_right_image",
                    #     "camera/image/varied_camera_1_left_image", "camera/image/varied_camera_1_right_image",
                    #     "camera/image/varied_camera_2_left_image", "camera/image/varied_camera_2_right_image",
                    # ],
                ],
                value_names=[
                    # "wrist",
                    # "wrist-stereo",
                    "3cams",
                    # "3cams-stereo",
                ]
            )
            generator.add_param(
                key="observation.modalities.obs.low_dim",
                name="ldkeys",
                group=2498,
                values=[
                    ["cartesian_position", "gripper_position"],
                    # [
                    #     "robot_state/cartesian_position", "robot_state/gripper_position",
                    #     "camera/extrinsics/hand_camera_left", "camera/extrinsics/hand_camera_left_gripper_offset",
                    #     "camera/extrinsics/hand_camera_right", "camera/extrinsics/hand_camera_right_gripper_offset",
                    #     "camera/extrinsics/varied_camera_1_left", "camera/extrinsics/varied_camera_1_right",
                    #     "camera/extrinsics/varied_camera_2_left", "camera/extrinsics/varied_camera_2_right",
                    # ]
                ],
                value_names=[
                    "proprio",
                    # "proprio-extrinsics",
                ]
            )

        else:
            raise ValueError
        generator.add_param(
            key="observation.encoder.rgb.core_kwargs.backbone_class",
            name="backbone",
            group=1234,
            values=[
                "ResNet18Conv",
                # "ResNet50Conv",
            ],
        )
        generator.add_param(
            key="observation.encoder.rgb.core_kwargs.feature_dimension",
            name="visdim",
            group=1234,
            values=[
                64,
                # 512,
            ],
        )

    elif args.env == "kitchen":
        generator.add_param(
            key="train.data",
            name="ds",
            group=2,
            values=[
                # [{"path": "~/datasets/kitchen/prior/human_demos/pnp_table_to_cab/bowls/20230816_im84.hdf5", "filter_key": "100_demos"}],
                [{"path": "~/datasets/kitchen/prior/human_demos/pnp_table_to_cab/all/20230806_im84.hdf5", "filter_key": "100_demos"}],
                # [{"path": "~/datasets/kitchen/prior/mimicgen/pnp_table_to_cab/viraj_mg_2023-08-10-20-31-14/demo_im84.hdf5", "filter_key": "100_demos"}],
                # [{"path": "~/datasets/kitchen/prior/mimicgen/pnp_table_to_cab/viraj_mg_2023-08-10-20-31-14/demo_im84.hdf5", "filter_key": "1000_demos"}],
            ],
            value_names=[
                # "bowls-human-100",
                "human-100",
                # "mg-100",
                # "mg-1000",
            ],
        )
        
        # update env config to use absolute action control
        generator.add_param(
            key="experiment.env_meta_update_dict",
            name="",
            group=-1,
            values=[
                {"env_kwargs": {"controller_configs": {"control_delta": False}}}
            ],
        )

        generator.add_param(
            key="train.action_keys",
            name="ac_keys",
            group=-1,
            values=[
                [
                    "action_dict/abs_pos",
                    "action_dict/abs_rot_6d",
                    "action_dict/gripper",
                    "action_dict/base_mode",
                    # "actions",
                ],
            ],
            value_names=[
                "abs",
            ],
            hidename=True,
        )
    elif args.env == "square":
        if args.data_format == 'hdf5':
            generator.add_param(
                key="train.data",
                name="ds",
                group=2,
                values=[
                    [
                        {"path": "/iris/u/jyang27/dev/robomimic/datasets/square/ph/low_dim_v141.hdf5"}, # replace with your own path
                    ],
                ],
                value_names=[
                    "square",
                ],
            )
        elif args.data_format == 'rlds':
            generator.add_param(
                key="train.data",
                name="ds",
                group=2,
                values=[
                    [
                        {"path": "/iris/u/jyang27/rlds_data",
                        "name": "robomimic_dataset"}, # replace with your own path
                    ],
                ],
                value_names=[
                    "square",
                ],
            )
        else:
            raise ValueError 

        # update env config to use absolute action control
        generator.add_param(
            key="experiment.env_meta_update_dict",
            name="",
            group=-1,
            values=[
                {"env_kwargs": {"controller_configs": {"control_delta": False}}}
            ],
        )
        
        generator.add_param(
            key="train.action_keys",
            name="ac_keys",
            group=-1,
            values=[
                [
                    "action_dict/abs_pos",
                    "action_dict/abs_rot_6d",
                    "action_dict/gripper",
                    # "actions",
                ],
            ],
            value_names=[
                "abs",
            ],
        )


    else:
        raise ValueError
    
    generator.add_param(
        key="train.output_dir",
        name="",
        group=-1,
        values=[
            "/iris/u/jyang27/expdata/{env}/{mod}/{algo_name_short}".format(
                env=args.env,
                mod=args.mod,
                algo_name_short=algo_name_short,
            )
        ],
    )

    return generator

if __name__ == "__main__":
    parser = get_argparser()

    args = parser.parse_args()
    make_generator(args, make_generator_helper)
