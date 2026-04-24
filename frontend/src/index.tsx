import React from "react";
import { views } from "@apache-superset/core";

const viewDisposable = views.registerView(
  { id: "pandas.semantic-layer", name: "Pandas Semantic Layer" },
  "sqllab.panels",
  () => <p>Pandas Semantic Layer</p>
);

export const activate = () => {
  console.log("Pandas Semantic Layer extension activated");
};

export const deactivate = () => {
  viewDisposable.dispose();
  console.log("Pandas Semantic Layer extension deactivated");
};
