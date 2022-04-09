from shutil import copy
from tempfile import NamedTemporaryFile

import numpy as np
from keras import Model
from selenium.webdriver import Keys
from tensorflow import keras

from time import sleep

from cv2 import cv2
from numpy import ndarray
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support.expected_conditions import presence_of_element_located
from selenium.webdriver.support.wait import WebDriverWait

from image_loading import encode_single_sample, char_to_num
from preprocessing import preprocess


def main():
    model: Model = keras.models.load_model('./model.h5')

    driver = webdriver.Chrome()
    while True:
        try:
            driver.get(
                "https://oss.uredjenazemlja.hr/public/cadServices.jsp?action=publicCadastreParcel&institutionID=32&cadastreMunicipalityId=2840&possessionSheetNr=907")
            pregledaj_pl_bzp_button = WebDriverWait(driver, 30).until(
                presence_of_element_located((
                    By.XPATH,
                    "/html/body/div/div/div[2]/div/div/div[2]/div[1]/div/div/div[1]/div[2]/div[1]/div/table/tbody/tr/td[3]/table/tbody/tr/td[2]/em/button"
                ))
            )
            pregledaj_pl_bzp_button.click()

            WebDriverWait(driver, 30).until(
                presence_of_element_located((
                    By.XPATH,
                    "/html/body/div/div/div[2]/div[2]/div[2]/div[1]/div/div/div[1]"
                ))
            )
            try:
                driver.execute_script(
                    'document.getElementsByClassName("gwt-Image")[0].parentElement.parentElement.parentElement.parentElement.parentElement.parentElement.parentElement.parentElement.parentElement.parentElement.style.width = "400px";')
                driver.execute_script(
                    'document.getElementsByClassName("gwt-Image")[0].parentElement.parentElement.parentElement.parentElement.parentElement.style.width = "400px";')
                driver.execute_script('document.getElementsByClassName("gwt-Image")[0].style.width = "276px";')

                kaptcha: WebElement = WebDriverWait(driver, 30).until(
                    presence_of_element_located((
                        By.CLASS_NAME,
                        "gwt-Image"
                    ))
                )

                kaptcha_file = NamedTemporaryFile(suffix=".png", delete=False)
                kaptcha_clean_file = NamedTemporaryFile(suffix=".png", delete=False)
                kaptcha_file.write(kaptcha.screenshot_as_png)
                kaptcha_file.close()
                kaptcha_clean_file.write(bytes([0]))
                kaptcha_clean_file.close()

                cv2.imwrite(kaptcha_clean_file.name, preprocess(cv2.imread(kaptcha_file.name, 0)))

                img, _ = encode_single_sample(kaptcha_clean_file.name, "xxxxx", True)
                img = np.expand_dims(img, axis=0)

                y_pred = model.predict(img)
                y_pred = np.argmax(y_pred, axis=2)
                num_to_char = {v: k for k, v in char_to_num.items()}
                num_to_char[-1] = 'UKN'
                solution = "".join(list(map(lambda x: num_to_char[x], y_pred[0])))

                input_field: WebElement = WebDriverWait(driver, 30).until(
                    presence_of_element_located((
                        By.XPATH,
                        "/html/body/div/div/div[2]/div[2]/div[2]/div[1]/div/div/div[1]/div/div[2]/div[1]/form/div[2]/div[1]/input"
                    ))
                )
                input_field.send_keys(solution)
                sleep(1)  # Kao eticki
                input_field.send_keys(Keys.ENTER)

                # noinspection PyBroadException
                try:
                    posjedovni_list_field: WebElement = WebDriverWait(driver, 3).until(
                        presence_of_element_located((
                            By.XPATH,
                            "/html/body/div/div/div[2]/div[2]/div[2]/div[1]/div/div/div[1]/div[1]/div/div/table/tbody/tr[16]/td[2]/span"
                        ))
                        # Pregledaj PL/BZP button
                    )
                    copy(kaptcha_file.name, f"./dataset_automatic/{solution}.png")
                except Exception:
                    copy(kaptcha_file.name, f"./bad_predictions/{solution}.png")
            except Exception:
                sleep(30)
        except Exception:
            sleep(30)

        sleep(5)  # Kao eticki


if __name__ == '__main__':
    main()
