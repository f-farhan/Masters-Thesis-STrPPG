import configparser
import argparse

def get_config():
    """
    Get the configuration in the config.ini file.
    The Default section are the default value of every parameter.
    """
    config = configparser.ConfigParser()
    config.read("config.ini")
    return config

def get_args():
    """
    Get the argument pass as command line argument.
    Default value are the one in config.ini but can be override with command line option
    ex : python Main.py --DEBUG=0
    set DEBUG to zero whereas it is 1 in the config.ini file
    """
    config = get_config()
    parser = argparse.ArgumentParser(description="Set the parameter of the script")
    parser.add_argument("--DEBUG"            , type=int, default=config.getint("DEFAULT","debug"), help="Save data to .npy file")
    parser.add_argument("--FORMAT"           , type=int, default=config.getint("DEFAULT","format"), help="Filtre bilateral 0=Uint8, 1=Uint16")
    parser.add_argument("--RESIZE"           , type=int, default=config.getint("DEFAULT","resize"), help="Downscale l'image celon ce facteur")
    parser.add_argument("--SHORTFRQ"         , type=int, default=config.getint("DEFAULT","shortfrq"), help="Short Filter      : 0 = Non, 1 = Oui ")
    parser.add_argument("--NORMALISATION"    , type=int, default=config.getint("DEFAULT","normalisation"), help="Normalisation     : 0 = Non, 1 = Oui")
    parser.add_argument("--DETREND"          , type=int, default=config.getint("DEFAULT","detrend"), help="Detrend           : 0 = Non, 1 = Oui")
    parser.add_argument("--REMOVE_BACKGROUND", type=int, default=config.getint("DEFAULT","remove_background"), help="Remove Background : 0 = Non, 1 = Oui")
    parser.add_argument("--METHOD",
                        type=str,
                        choices=["Green", "POS", "Chrom", "G-R"],
                        default=config.get("DEFAULT", "method").replace("\"",""),
                        help="Choix de la methode de filtrage")
    parser.add_argument("--WINLENGTHSEC"     , type=int, default=config.getint("DEFAULT","winlengthsec"), help="Taille de la fenetre glissante")
    parser.add_argument("--GPU"              , type=int, default=config.getint("DEFAULT","gpu"), help="Utilisation du GPU : 0 = Non, 1 = Oui")
    parser.add_argument("--HEATMAP"          , type=int, default=config.getint("DEFAULT","heatmap"), help="Creer une Heatmap: 0 = Non, 1 = Oui")
    parser.add_argument("--AFFICHAGE"        , type=int, default=config.getint("DEFAULT","affichage"), help="Affichage des plots : 0 = Non, 1 = Oui")
    parser.add_argument("--NBPIXELS"         , type=int, default=config.getint("DEFAULT","nbpixels"), help="Save data to .npy file")
    parser.add_argument("--SELECT_ROI"       , type=int, default=config.getint("DEFAULT","select_roi"), help="ROI pour calculer perfusion map, 0 = calcul toute la frame")
    parser.add_argument("--SELECT_ROI_REF"   , type=int, default=config.getint("DEFAULT","select_roi_ref"), help="ROI pour estimer signal de reference, 0 = alors reference = signal moyen sur toute la frame")
    args = parser.parse_args()
    return args


class Config:
    """
    Singleton class ensures there is only one config object during the run.
    Since it is called in nearly every function it would be a waste of ressources
    to create a new one for every function call.

    Properties are here to make the configuration read-only,
    since user should no be able to modify it during the run.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        self._cf = get_args()

    @property
    def DEBUG            (self):
        return self._cf.DEBUG
    @property     
    def FORMAT           (self):
        return self._cf.FORMAT
    @property     
    def RESIZE           (self):
        return self._cf.RESIZE
    @property     
    def SHORTFRQ         (self):
        return self._cf.SHORTFRQ
    @property     
    def NORMALISATION    (self):
        return self._cf.NORMALISATION
    @property     
    def DETREND          (self):
        return self._cf.DETREND
    @property     
    def REMOVE_BACKGROUND(self):
        return self._cf.REMOVE_BACKGROUND
    @property     
    def FAKEVID          (self):
        return self._cf.FAKEVID
    @property     
    def METHOD           (self):
        return self._cf.METHOD
    @property     
    def WINLENGTHSEC     (self):
        return self._cf.WINLENGTHSEC
    @property     
    def GPU              (self):
        return self._cf.GPU
    @property     
    def HEATMAP          (self):
        return self._cf.HEATMAP
    @property     
    def AFFICHAGE        (self):
        return self._cf.AFFICHAGE
    @property     
    def NBPIXELS         (self):
        return self._cf.NBPIXELS
    @property     
    def SELECT_ROI       (self):
        return self._cf.SELECT_ROI
    @property     
    def SELECT_ROI_REF   (self):
        return self._cf.SELECT_ROI_REF

cf = Config()




