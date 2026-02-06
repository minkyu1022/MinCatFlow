document.addEventListener("DOMContentLoaded", function () {
  var stage = new NGL.Stage("viewport-sample", { backgroundColor: "white" });

  var reps = {
    base: null,
    unitcell: null,
    supercell: null
  };

  stage.loadFile("data/relaxed_sample.pdb", {
    defaultRepresentation: false,
    ext: "pdb",
  }).then(function (comp) {
    comp.setName("catalyst-sample");

    reps.base = comp.addRepresentation("spacefill", {
      colorScheme: "element",
      radiusScale: 0.5,
      visible: true
    });

    reps.unitcell = comp.addRepresentation("unitcell", {
      colorValue: "gray",
      radiusScale: 1.0,
      visible: false
    });

    reps.supercell = comp.addRepresentation("spacefill", {
      colorScheme: "element",
      radiusScale: 0.5,
      assembly: "SUPERCELL",
      visible: false
    });

    var assembly = new NGL.Assembly("mySupercell");
    var vectors = [
      [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]
    ];

    var box = comp.structure.unitcell;

    if (box) {
      vectors.forEach(function (v) {
        var shift = new NGL.Vector3();
        shift.add(new NGL.Vector3().copy(box.getBasisVector(0)).multiplyScalar(v[0]));
        shift.add(new NGL.Vector3().copy(box.getBasisVector(1)).multiplyScalar(v[1]));
        shift.add(new NGL.Vector3().copy(box.getBasisVector(2)).multiplyScalar(v[2]));

        var matrix = new NGL.Matrix4().makeTranslation(shift.x, shift.y, shift.z);
        assembly.addPart([0, 1, 2, 3, 4, 5], matrix);
      });

      comp.structure.biomolDict.mySupercell = assembly;

      reps.supercell = comp.addRepresentation("spacefill", {
        colorScheme: "element",
        radiusScale: 0.5,
        assembly: "mySupercell",
        visible: false
      });
    }

    comp.autoView();

    // Spin
    var toggleSpinBtn = document.getElementById("toggleSpin-sample");
    var isSpinning = false;
    if (toggleSpinBtn) {
      toggleSpinBtn.addEventListener("click", function () {
        if (!isSpinning) {
          stage.setSpin([0, 1, 0], 0.01);
          isSpinning = true;
          toggleSpinBtn.textContent = "Stop Spin";
        } else {
          stage.setSpin(null, null);
          isSpinning = false;
          toggleSpinBtn.textContent = "Spin";
        }
      });
    }

    // Unit Cell
    var toggleCellBtn = document.getElementById("toggleCell-sample");
    if (toggleCellBtn) {
      toggleCellBtn.addEventListener("click", function () {
        if (reps.unitcell) {
          var isVisible = !reps.unitcell.visible;
          reps.unitcell.setVisibility(isVisible);

          toggleCellBtn.style.fontWeight = isVisible ? "bold" : "normal";
          toggleCellBtn.style.color = isVisible ? "blue" : "black";
        }
      });
    }

    // Supercell
    var toggleSupercellBtn = document.getElementById("toggleSupercell-sample");
    if (toggleSupercellBtn) {
      toggleSupercellBtn.addEventListener("click", function () {
        if (reps.supercell && reps.base) {
          var isSupercellOn = !reps.supercell.visible;

          reps.supercell.setVisibility(isSupercellOn);
          reps.base.setVisibility(!isSupercellOn);

          toggleSupercellBtn.style.fontWeight = isSupercellOn ? "bold" : "normal";
          toggleSupercellBtn.style.color = isSupercellOn ? "blue" : "black";

          if (isSupercellOn) comp.autoView(500);
        }
      });
    }

    // Reset View
    var resetViewBtn = document.getElementById("resetView-sample");
    if (resetViewBtn) {
      resetViewBtn.addEventListener("click", function () {
        comp.autoView(500);
      });
    }
  });

  window.addEventListener("resize", function () {
    stage.handleResize();
  });
});