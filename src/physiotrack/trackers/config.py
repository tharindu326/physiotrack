import os 


class Config:
    """Single configuration class for all Tracker settings"""
    def __init__(self):
        # General settings
        self.device = 'cuda'
        self.colors = {
            'blue': (255, 0, 0),
            'red': (0, 0, 255),
            'green': (0, 255, 0),
            'yellow': (0, 255, 255),
            'purple': (255, 0, 255)
        }
        self.text_size = 1
        
        # Overlay options
        self.show_detection_boxes = False  # Red boxes for raw detections
        self.show_original_tracks = False   # Green boxes for all MOT tracks
        self.show_student_track = True     # Blue box for isolated student track
        self.show_tracking_tail = True     # Blue trail for student movement
        self.show_all_trails = False       # Show trails for all tracks (not just student)
        self.tail_opacity = 0.7           # Opacity of tracking tail
        self.debug_mode = False
        
        # Tracker settings
        self.tracker_type = 'ocsort'  # Options: 'bytetrack', 'strongsort', 'ocsort', 'boosttrack'
        self.classes = [0]  # Classes to track
        self.trail_length = 30
        
        # Student tracking settings
        self.enable_student_tracking = False
        self.required_consecutive_frames = 30
        self.inconsistent_motion_threshold = 5
        self.student_reinit_iou_threshold = 0.3
        
        # ByteTrack settings
        self.bytetrack_track_thresh = 0.25
        self.bytetrack_match_thresh = 0.8
        self.bytetrack_track_buffer = 30
        self.bytetrack_frame_rate = 30
        
        # StrongSORT settings
        model_dir = os.path.join(os.path.dirname(__file__), '..', 'model_data')
        self.strongsort_reid_weights = os.path.join(model_dir, 'osnet_x0_25_msmt17.pt')
        self.strongsort_max_dist = 0.2
        self.strongsort_max_iou_dist = 0.7
        self.strongsort_max_age = 70
        self.strongsort_max_unmatched_preds = 7
        self.strongsort_n_init = 3
        self.strongsort_nn_budget = 100
        self.strongsort_mc_lambda = 0.995
        self.strongsort_ema_alpha = 0.9
        
        # OCSort settings
        self.ocsort_det_thresh = 0.2
        self.ocsort_max_age = 30
        self.ocsort_min_hits = 3
        self.ocsort_iou_thresh = 0.3
        self.ocsort_delta_t = 3
        self.ocsort_asso_func = 'iou'
        self.ocsort_inertia = 0.2
        self.ocsort_use_byte = False
        
        # BoostTrack settings
        self.boosttrack_det_thresh = 0.2
        self.boosttrack_lambda_iou = 0.5
        self.boosttrack_lambda_mhd = 0.5
        self.boosttrack_lambda_shape = 0.5
        self.boosttrack_dlo_boost_coef = 0.9
        self.boosttrack_use_dlo_boost = True
        self.boosttrack_use_duo_boost = True
        self.boosttrack_max_age = 30

    def print(self):
        """Print all configuration settings in an organized format."""
        print("\n" + "="*60)
        print(" TRACKER CONFIGURATION ")
        print("="*60)
        
        # Group configs by category
        categories = {
            "General Settings": ["device", "text_size", "tracker_type", "classes", "trail_length"],
            "Student Tracking": ["enable_student_tracking", "required_consecutive_frames", 
                               "inconsistent_motion_threshold", "student_reinit_iou_threshold"],
            "Overlay Options": ["show_detection_boxes", "show_original_tracks", "show_student_track",
                              "show_tracking_tail", "show_all_trails", "tail_opacity"],
            "ByteTrack": [k for k in vars(self) if k.startswith("bytetrack_")],
            "StrongSORT": [k for k in vars(self) if k.startswith("strongsort_")],
            "OCSort": [k for k in vars(self) if k.startswith("ocsort_")],
            "BoostTrack": [k for k in vars(self) if k.startswith("boosttrack_")]
        }
        
        for category, keys in categories.items():
            print(f"\n{category}:")
            for key in keys:
                if hasattr(self, key):
                    value = getattr(self, key)
                    # Format the key name for display
                    display_key = key.replace("_", " ").title()
                    print(f"  {display_key:<35} : {value}")
        
        print("\n" + "="*60 + "\n")